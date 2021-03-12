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

def test_data_gen(sensors=(
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
        ),
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
        "lidar"), path=None, outpath=None, token=None, start=None, end=None, step=None):
    if path is None:
        path=os.path.join(DATA_PATH, "rgb")
    if token is None:
        token = random.choice(os.listdir(path))
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "visualization", "rgb")

    output_dirs = {sensor: os.path.join(outpath, sensor) for sensor in sensors}
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    episode = Episode(path, token)
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
                plt.imsave(os.path.join(output_dir, "{a}{i}.png".format(a=sensor,i=i)), img)


def imgs_to_gif(inpath, outfile, prefix, start=0, end=None):
    count = len(os.listdir(inpath))
    if end is None:
        end = count
    assert start <= end <= count
    images = []
    for i in tqdm.trange(start, end):
        filename = os.path.join(inpath, "{prefix}{i}.png".format(prefix=prefix,i=i))
        images.append(imageio.imread(filename))    
    imageio.mimsave(outfile, images)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "lidar"), os.path.join(DATA_PATH, "visualization", "rgb", "lidar.gif"), "c", 100, 200)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb"), os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb.gif"), "front_camera_rgb", 100, 400)


def generate_distributions(root_path=None,sensors=None,n_frames=2000,n_episodes=20,
        weathers=None, n_ped_cars=None,towns=None,skip=0):
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
        n_ped_cars = (0, 1000)
    if towns is None:
        towns = ("Town01", "Town02")
    for weather, n, town, i in tqdm.tqdm(list(itertools.product(weathers, n_ped_cars, towns, range(n_episodes)))[skip:]):
        path = os.path.join(root_path, town+weather+str(n))
        collect_not_moving_counts(town, path, n, n, n_frames, sensors, agent_fn, weather)

def collect_not_moving_counts(town, output_dir, num_vehicles, num_pedestrains, n_frames, sensors, agent_fn, weather):
    while True:
        os.makedirs(output_dir, exist_ok=True)
        listdir = os.listdir(output_dir)
        CARLADataset.collect(town, output_dir, num_vehicles, num_pedestrains, n_frames, None, None, sensors, False, agent_fn, weather)
        newdir = [x for x in os.listdir(output_dir) if x not in listdir][0]  # find new folder
        newdir = os.path.join(output_dir, newdir)
        counts = CARLADataset.car_not_moving_counts(newdir)
        print(counts)
        if 0.5*sum(counts) < n_frames and counts[-1] < 0.2*n_frames:
            break
        shutil.rmtree(newdir)
        print("repeat",weather, num_vehicles, town)


def process_distributions(inpath=None, outpath=None, num_frame_skips=5):
    if inpath is None:
        inpath = os.path.join(DATA_PATH, "dists","raw", "train")
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "dists", "processed","train")
    for dist in os.listdir(inpath):
        CARLADataset.process(os.path.join(inpath, dist), os.path.join(outpath, dist), num_frame_skips=num_frame_skips)

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

if __name__=="__main__":
    if False:
        ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d128", "ckpts","model-108.pt")
        data_path = os.path.join(DATA_PATH, "dists3", "raw", "test")
        arff_path = os.path.join(DATA_PATH, "dists3", "raw", "test.arff")
        model = getDIM(ckpt_path,device,128)
        mobilenet = dict(model.named_children())["_encoder"]
        modalities = (
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "velocity",
        )

        CARLADataset.annotate_with_model(data_path, modalities, mobilenet, "mobilenet_d128_e108",device=device)
        CARLADataset.make_arff(data_path, arff_path,("mobilenet_d128_e108",),"mobilenet_d128_e108")
    if True:
        root_path = os.path.join(DATA_PATH,"dists4","raw","test")
        d0 = ["Town01","ClearNoon",0]
        d1 = ["Town02","HardRainNoon",1000]
        dists = [d0,d0,d1,d1,d0,d0,d1,d1]
        n_frames=2000
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
            path = os.path.join(root_path, str(i)+"_"+town+weather+str(n))
            collect_not_moving_counts(town, path, n, n, n_frames, sensors, agent_fn, weather)
    if False:
        inpath = os.path.join(DATA_PATH, "dists2","raw", "train")
        outpath5 = os.path.join(DATA_PATH, "dists2", "processed5","train")
        outpath1 = os.path.join(DATA_PATH, "dists2", "processed1","train")
        process_distributions(inpath,outpath5,num_frame_skips=5)
        process_distributions(inpath,outpath1,num_frame_skips=1)
    
