from defaults import PATH, MODELS_PATH, DATA_PATH, device
from evaluate import getDIM

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import imageio
import glob
import torch
import shutil

import oatomobile
import oatomobile.envs
from oatomobile.datasets.carla import CARLADataset, get_npz_files
import oatomobile.baselines.torch.dim.train as train
from oatomobile.core.dataset import Episode
from oatomobile.baselines.rulebased.autopilot.agent import AutopilotAgent
from oatomobile.torch.networks.perception import MobileNetV2
from oatomobile.baselines.torch.dim.model import ImitativeModel
from oatomobile.baselines.torch.dim.agent import DIMAgent
import carla
import itertools

from evaluate import getDIM, get_agent_fn

WEATHERS = {
    "ClearNoon":carla.WeatherParameters.ClearNoon,
    "ClearSunset":carla.WeatherParameters.ClearSunset,
    "CloudyNoon":carla.WeatherParameters.CloudyNoon,
    "CloudySunset":carla.WeatherParameters.CloudySunset,
    "Default":carla.WeatherParameters.Default,
    "HardRainNoon":carla.WeatherParameters.HardRainNoon,
    "HardRainSunset":carla.WeatherParameters.HardRainSunset,
    "MidRainSunset":carla.WeatherParameters.MidRainSunset,
    "MidRainyNoon":carla.WeatherParameters.MidRainyNoon,
    "SoftRainNoon":carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset":carla.WeatherParameters.SoftRainSunset,
    "WetCloudyNoon":carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset":carla.WeatherParameters.WetCloudySunset,
    "WetNoon":carla.WeatherParameters.WetNoon,
    "WetSunset":carla.WeatherParameters.WetSunset
}

ALL_SENSORS = (
        "acceleration",
        "velocity",
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "actors_tracker",
        "front_camera_rgb",
        "rear_camera_rgb",
        "left_camera_rgb",
        "right_camera_rgb",
        "bird_view_camera_rgb",
        "bird_view_camera_cityscapes",
        )

def test_data_gen(sensors=ALL_SENSORS,
        agent_fn=AutopilotAgent,
        path=os.path.join(DATA_PATH, "dim")):
    """

    """
    CARLADataset.collect("Town01", path, 100, 100, 1000, None, None, sensors, False, agent_fn)


def download():
    """ download the data from oatomobile
    """
    raw = CARLADataset("raw")
    examples = CARLADataset("examples")
    processed = CARLADataset("processed")
    datasets = [raw, examples, processed]
    for dataset in datasets:
        dataset.download_and_prepare(os.path.join(DATA_PATH, "downloaded"))


def process_downloaded():
    CARLADataset.process(os.path.join(DATA_PATH,"downloaded","raw"),os.path.join(DATA_PATH,"downloaded","selfprocessed"))


def visualize():
    examples_path = os.path.join(DATA_PATH,"downloaded","examples","train")
    examples_file = os.path.join(examples_path,random.choice(os.listdir(examples_path)))

    raw_path = os.path.join(DATA_PATH,"downloaded","raw")
    raw_path = os.path.join(raw_path, random.choice(os.listdir(raw_path)))
    raw_file = os.path.join(raw_path, random.choice(os.listdir(raw_path)))

    processed_path = os.path.join(DATA_PATH,"downloaded","processed","train")
    processed_file = os.path.join(processed_path,random.choice(os.listdir(processed_path)))

    files = [examples_file, raw_file, processed_file]
    output_dir = os.path.join(DATA_PATH, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    output_dirs = [os.path.join(output_dir, x) for x in ["examples", "raw", "processed"]]
    for out, file in zip(output_dirs, files):
        CARLADataset.plot_datum(file, out)


def visualize_raw_lidar(path=None, outpath=None, start=None, end=None, step=None):
    if path is None:
        path = os.path.join(DATA_PATH, "rgb")
        path = os.path.join(path, random.choice(os.listdir(path)))
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "visualization", "rgb")
    output_dir1 = os.path.join(outpath, "lidar1")
    output_dir2 = os.path.join(outpath, "lidar2")
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    output_dir = os.path.join(outpath, "lidar")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(path, "metadata")) as metadata:
        for i, line in enumerate(tqdm.tqdm(metadata.readlines()[start:end:step])):
            line = line[:-1] + ".npz"
            x = np.load(os.path.join(path, line))
            lidar = x.f.lidar
            plt.imsave(os.path.join(output_dir1, "a{i}.png".format(i=i)), lidar[:,:,0], cmap="gray")
            plt.imsave(os.path.join(output_dir2, "b{i}.png".format(i=i)), lidar[:,:,1], cmap="gray")
            img = np.zeros(lidar.shape[:2]+(3,))
            img[:,:,:2] = lidar
            plt.imsave(os.path.join(output_dir, "c{i}.png".format(i=i)), img)


        # visualize_raw_rgb(path=os.path.join(DATA_PATH, "dists", "Town01HardRainNoon1000"),outpath=os.path.join(DATA_PATH, "vis", "Town01HardRainNoon1000"), end=1000, step=10)
        # visualize_raw_rgb(path=os.path.join(DATA_PATH, "dists", "Town02HardRainNoon0"),outpath=os.path.join(DATA_PATH, "vis", "Town02HardRainNoon0"), end=1000, step=10)
def visualize_raw_rgb(sensors=(
        "front_camera_rgb",
        "rear_camera_rgb",
        "left_camera_rgb",
        "right_camera_rgb",
        "bird_view_camera_rgb",
        "bird_view_camera_cityscapes",
        "lidar"), path=None, outpath=None, start=None, end=None, step=None):
    if path is None:
        path=os.path.join(DATA_PATH, "rgb")
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "visualization", "rgb")

    output_dirs = {sensor: os.path.join(outpath, sensor) for sensor in sensors}
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    episode = Episode(path, "")
    for i, line in enumerate(tqdm.tqdm(episode.fetch()[start:end:step])):
        x = episode.read_sample(line, None)
        for sensor, output_dir in output_dirs.items():
            if sensor in x:
                img = x[sensor]
                if sensor == "lidar":
                    img2 = np.zeros(img.shape[:2]+(3,))
                    img2[:,:,:2] = img
                    img = img2
                    pass
                plt.imsave(os.path.join(output_dir, "{a}_{i}.png".format(a=sensor,i=i)), img)


def imgs_to_gif(inpath, outfile, prefix, start=0, end=None):
    count = len(os.listdir(inpath))
    if end is None:
        end = count
    assert start <= end <= count
    images = []
    for i in tqdm.trange(start, end):
        filename = os.path.join(inpath, "{prefix}_{i}.png".format(prefix=prefix,i=i))
        images.append(imageio.imread(filename))    
    imageio.mimsave(outfile, images)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "lidar"), os.path.join(DATA_PATH, "visualization", "rgb", "lidar.gif"), "c", 100, 200)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb"), os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb.gif"), "front_camera_rgb", 100, 400)


def generate_distributions(root_path=None,sensors=None,n_frames=2000,n_episodes=20,
        weathers=None, n_ped_cars=None,towns=None,start=0,end=None):
    if root_path is None:
        root_path = os.path.join(DATA_PATH, "dists2", "train")
    if sensors is None:
        sensors = (
            "acceleration",
            "velocity",
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "actors_tracker",
        )
    agent_fn=AutopilotAgent
    if weathers is None:
        weathers = ("HardRainNoon", "ClearNoon")
    if n_ped_cars is None:
        n_ped_cars = (0, 50)
    if towns is None:
        towns = ("Town01", "Town03")
    for weather, n, town, i in tqdm.tqdm(list(itertools.product(weathers, n_ped_cars, towns, range(n_episodes)))[start:end]):
        path = os.path.join(root_path, town+weather+str(n))
        collect_not_moving_counts(town, path, n, n, n_frames, sensors, agent_fn, weather)

def collect_not_moving_counts(town, output_dir, num_vehicles, num_pedestrains, n_frames, sensors, agent_fn, weather,visualize=True,max_not_moving=0.5,max_not_moving_end=0.4):
    while True:
        os.makedirs(output_dir, exist_ok=True)
        listdir = os.listdir(output_dir)
        CARLADataset.collect(town, output_dir, num_vehicles, num_pedestrains, n_frames, None, None, sensors, False, agent_fn, weather)
        newdir = [x for x in os.listdir(output_dir) if x not in listdir][0]  # find new folder
        newdir_path = os.path.join(output_dir, newdir)
        counts = CARLADataset.car_not_moving_counts(newdir_path)
        print(counts)
        if np.sum(counts) > max_not_moving*n_frames or counts[-1] > max_not_moving_end*n_frames:
            shutil.rmtree(newdir_path)
            print("repeat",weather, num_vehicles, town)
            continue
        break

    if visualize:
        vis_path = os.path.join(output_dir,"vis",newdir)
        os.makedirs(vis_path, exist_ok=True)
        with open(os.path.join(vis_path, "not_moving_counts.txt"),"w") as counts_file:
            counts_file.write(str(counts))
        visualize_raw_rgb(path=newdir_path,outpath=vis_path,)

def process_distributions(inpath=None, outpath=None, num_frame_skips=5,min_distance_since_last=0.01,min_distance_trajectory=0.01):
    if inpath is None:
        inpath = os.path.join(DATA_PATH, "dists","raw", "train")
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "dists", "processed","train")
    for dist in os.listdir(inpath):
        CARLADataset.process(os.path.join(inpath, dist), os.path.join(outpath, dist),num_frame_skips=num_frame_skips,
        min_distance_since_last=min_distance_since_last,min_distance_trajectory=min_distance_trajectory)

def test_annotate_no_corruption():
    orig_path = os.path.join(DATA_PATH, "dists3", "raw", "0_Town01ClearNoon0","b78cf9653f3b4c4ab69f2fc60b4e05d1")
    mod_path = os.path.join(DATA_PATH, "dists3", "raw","test", "0_Town01ClearNoon0","b78cf9653f3b4c4ab69f2fc60b4e05d1")
    for f in os.listdir(orig_path):
        if f.endswith(".npz"):
            orig_file = dict(np.load(os.path.join(orig_path, f)))
            mod_file = dict(np.load(os.path.join(mod_path, f)))
            for key in orig_file:
                comp = orig_file[key]==mod_file[key]
                if isinstance(comp, np.ndarray) and not comp.all() or not isinstance(comp, np.ndarray) and not comp:
                    print(orig_file[key],)
                    print(mod_file[key])
                    print()

def test_get_npz(path,outpath):
    os.makedirs(outpath,exist_ok=True)
    files = get_npz_files(path)
    for i in range(len(files)):
        x = dict(np.load(files[i]))
        img = x["front_camera_rgb"]
        plt.imsave(os.path.join(outpath,str(i)+".png"),img)


if __name__=="__main__":
    if False:  # generate dists
        root_path = os.path.join(DATA_PATH, "dists10", "raw", "train",)
        generate_distributions(root_path, n_frames=2000, n_episodes=100)
        root_path = os.path.join(DATA_PATH, "dists10", "raw", "val",)
        generate_distributions(root_path, n_frames=2000, n_episodes=5)
        # root_path = os.path.join(DATA_PATH, "dists7", "raw","train")
        # generate_distributions(root_path, n_frames=2000, n_episodes=50)

    if False:
        # raw_path = os.path.join(DATA_PATH, "dists7.2", "raw","train","train1")
        # processed_path = os.path.join(DATA_PATH, "dists7.2", "processed5_moving3","train","train1")
        # process_distributions(raw_path, processed_path,num_frame_skips=5,min_distance_since_last=0,min_distance_trajectory=20)
        # raw_path = os.path.join(DATA_PATH, "dists7.2", "raw","train","train2")
        # processed_path = os.path.join(DATA_PATH, "dists7.2", "processed5_moving3","train","train2")
        # process_distributions(raw_path, processed_path,num_frame_skips=5,min_distance_since_last=0,min_distance_trajectory=20)
        # raw_path = os.path.join(DATA_PATH, "dists7.2", "raw","val")
        # processed_path = os.path.join(DATA_PATH, "dists7.2", "processed5_moving3","val")
        # process_distributions(raw_path, processed_path,num_frame_skips=5,min_distance_since_last=0,min_distance_trajectory=20)

        raw_path = os.path.join(DATA_PATH, "dists10", "raw","train")
        processed_path = os.path.join(DATA_PATH, "dists10", "processed5_movingl0.01","train")
        process_distributions(raw_path, processed_path,num_frame_skips=5,min_distance_since_last=0.01,min_distance_trajectory=0)
        raw_path = os.path.join(DATA_PATH, "dists10", "raw","val")
        processed_path = os.path.join(DATA_PATH, "dists10", "processed5_movingl0.01","val")
        process_distributions(raw_path, processed_path,num_frame_skips=5,min_distance_since_last=0.01,min_distance_trajectory=0)

    if False:  # create test distributions
        root_path = os.path.join(DATA_PATH,"dists16","raw","test")
        d0 = ["Town01","ClearNoon",0]
        d1 = ["Town03","HardRainNoon",50]
        dist_list = [d0,d1]
        dists = [d0,d1,d0,d1,d0,d1,d0,d1,d0,d1,d0,d1,d0,d1,d0,d1]
        n_frames=1000
        os.makedirs(root_path, exist_ok=True)
        target = list(np.asarray([[dist_list.index(d)]*n_frames for d in dists]).flat)
        with open(os.path.join(root_path, "target_dists.txt"),"w") as f:
            f.write(str(target))
        sensors = (
                "acceleration",
                "velocity",
                "lidar",
                "is_at_traffic_light",
                "traffic_light_state",
                "actors_tracker",
                "front_camera_rgb",
            )
        agent_fn = AutopilotAgent
        for i, (town, weather, n) in enumerate(dists):
            if i in (1,3,15):
                path = os.path.join(root_path, str(i)+"_"+town+weather+str(n))
                collect_not_moving_counts(town, path, n, n, n_frames, sensors, agent_fn, weather,max_not_moving=0.1)


    if False:  # create arffs
        
        ckpt_path = os.path.join(MODELS_PATH, "dim","dists10_d32", "ckpts","model-180.pt")
        data_path = os.path.join(DATA_PATH, "dists16", "raw", "test")
        arff_path = os.path.join(DATA_PATH, "dists16", "raw", "dists16_mobilenet_dists10_d32_e180.arff")
        model = getDIM(ckpt_path,device,32)
        mobilenet = dict(model.named_children())["_encoder"]
        modalities = (
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "velocity",
        )
        CARLADataset.annotate_with_model(data_path, modalities, mobilenet, "mobilenet_dists10_d32_e180",device=device)
        CARLADataset.make_arff(data_path, arff_path,("mobilenet_dists10_d32_e180",),"dists16_mobilenet_dists10_d32_e180")
        

        # ckpt_path = os.path.join(MODELS_PATH, "dim","dists10_d32", "ckpts","model-180.pt")
        # data_path = os.path.join(DATA_PATH, "dists12", "raw", "test")
        # arff_path = os.path.join(DATA_PATH, "dists12", "raw", "dists12_mobilenet_dists10_d32_e180.arff")
        # model = getDIM(ckpt_path,device,32)
        # mobilenet = dict(model.named_children())["_encoder"]
        # modalities = (
        #     "lidar",
        #     "is_at_traffic_light",
        #     "traffic_light_state",
        #     "velocity",
        # )
        # CARLADataset.annotate_with_model(data_path, modalities, mobilenet, "mobilenet_dists10_d32_e180",device=device)
        # CARLADataset.make_arff(data_path, arff_path,("mobilenet_dists10_d32_e180",),"dists12_mobilenet_dists10_d32_e180")
        
    
    
    if False:
        root_path = os.path.join(DATA_PATH, "dists4", "raw", "test")
        root_outpath = os.path.join(DATA_PATH, "dists4", "raw", "test_vis")
        for dist in os.listdir(root_path):
            dist_path = os.path.join(root_path, dist)
            for episode in os.listdir(dist_path):
                episode_path = os.path.join(dist_path, episode)
                outpath = os.path.join(root_outpath, dist, episode)
                visualize_raw_rgb(path=episode_path, outpath=outpath)
    
    if False:  # get gifs
        sensor = "lidar"
        root_path = os.path.join(DATA_PATH, "dists6", "raw", "test")
        for dist in os.listdir(root_path):
            dist_path = os.path.join(root_path, dist, "vis")
            if os.path.isdir(dist_path):
                episode = os.listdir(dist_path)[0]
                episode_path = os.path.join(dist_path, episode)
                rgb_path = os.path.join(episode_path, sensor)
                gif_path = os.path.join(root_path, dist+"_"+sensor+".gif")
                imgs_to_gif(rgb_path, gif_path, sensor, start=1300, end=1400)

    if False:  # let model drive
        n_frames = 2000

        ckpt_path = os.path.join(MODELS_PATH, "dim","dists10_d32", "ckpts","model-144.pt")
        dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        agent_fn = get_agent_fn(dim)
        data_path = os.path.join(DATA_PATH, "test_dim3", "dists10_d32")
        collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

        # ckpt_path = os.path.join(MODELS_PATH, "dim","dists7.2_moving_d32", "ckpts","model-40.pt")
        # dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        # agent_fn = get_agent_fn(dim)
        # data_path = os.path.join(DATA_PATH, "test_dim3", "dists7.2_moving_d32")
        # collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

        # ckpt_path = os.path.join(MODELS_PATH, "dim","dists7.2_moving2_d32", "ckpts","model-100.pt")
        # dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        # agent_fn = get_agent_fn(dim)
        # data_path = os.path.join(DATA_PATH, "test_dim3", "dists7.2_moving2_d32")
        # collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

        # ckpt_path = os.path.join(MODELS_PATH, "dim","dists7.2_moving3_d32", "ckpts","model-100.pt")
        # dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        # agent_fn = get_agent_fn(dim)
        # data_path = os.path.join(DATA_PATH, "test_dim3", "dists7.2_moving3_d32")
        # collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

        # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d32", "ckpts","model-200.pt")
        # dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        # agent_fn = get_agent_fn(dim)
        # data_path = os.path.join(DATA_PATH, "test_dim3", "downloaded_d32")
        # collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

        # ckpt_path = None
        # dim=getDIM(ckpt_path,device=device,mobilenet_num_classes=32)
        # agent_fn = get_agent_fn(dim)
        # data_path = os.path.join(DATA_PATH, "test_dim3", "untrained")
        # collect_not_moving_counts("Town01", data_path,50,50,n_frames,ALL_SENSORS,agent_fn,"ClearNoon",True,1,1)

    if True:
        data_path = os.path.join(DATA_PATH, "dists13", "raw", "test")
        out_path = os.path.join(DATA_PATH, "dists13", "raw", "imgs")
        test_get_npz(data_path,out_path)
