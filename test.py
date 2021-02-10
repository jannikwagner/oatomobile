import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import imageio
import glob
import torch
from defaults import WEATHERS, PATH, DATA_PATH, MODELS_PATH

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

from load_model import getDIM

def save_imgs():
    os.chdir(DATA_PATH)
    if "imgs" not in os.listdir():
        os.mkdir("imgs")
    dir = random.choice(os.listdir())
    os.chdir(dir)
    for i, file in  enumerate(os.listdir()):
        x = np.load(file)
        lidar = x.f.lidar
        plt.imsave("../imgs/a{i}.png".format(i=i), lidar[:,:,1])
        plt.imsave("../imgs/b{i}.png".format(i=i), lidar[:,:,0])
        
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


def generate_distributions():
    sensors=(
        "acceleration",
        "velocity",
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "actors_tracker",
    )
    agent_fn=AutopilotAgent
    n_frames = 1000
    n_episodes = 2
    weathers = ("HardRainNoon", "ClearNoon")
    n_ped_cars = (0, 1000)
    towns = ("Town01", "Town02")
    skip = 0
    path = os.path.join(DATA_PATH, "dists", "raw", "val2")
    for weather, n, town, i in tqdm.tqdm(list(itertools.product(weathers, n_ped_cars, towns, range(n_episodes)))[skip:]):
        CARLADataset.collect(town, os.path.join(path, town+weather+str(n)), n, n, n_frames, None, None, sensors, False, agent_fn, carla.WeatherParameters.__dict__[weather])


def process_distributions(inpath=None, outpath=None):
    if inpath is None:
        inpath = os.path.join(DATA_PATH, "dists", "raw", "val")
    if outpath is None:
        outpath = os.path.join(DATA_PATH, "dists", "processed","val")
    for dist in os.listdir(inpath):
        if dist != os.path.split(outpath)[-1]:
            CARLADataset.process(os.path.join(inpath, dist), os.path.join(outpath, dist))


def test_collect():
    CARLADataset.collect("Town01", os.path.join(DATA_PATH, "test"), 0, 0, 50, None, None, ("lidar",), False, AutopilotAgent, None)
    CARLADataset.collect("Town01", os.path.join(DATA_PATH, "test"), 0, 0, 50, None, None, ("lidar",), False, AutopilotAgent, None)


if __name__=="__main__":
    if True:
        # model_path = os.path.join(MODELS_PATH, "dim", "dists2", "ckpts", "model-200.pt")
        # model = getDIM(model_path)
        # mobilenet = dict(model.named_children())["_encoder"]
        modalities = (
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "player_future",
            "velocity",
        )
        #CARLADataset.annotate_with_model(os.path.join(DATA_PATH, "dists","processed","train"), modalities, mobilenet, "mobilenet", None)
        print("annotated "+"#"*100+"\n\n")
        CARLADataset.make_arff(os.path.join(DATA_PATH, "dists","processed","train"), os.path.join(DATA_PATH, "dists","processed","dummy.arff"),("mobilenet", ),"mobilenet",recursive=True, num=100)
    #visualize_raw_rgb(("lidar",), os.path.join(DATA_PATH,"dists","processed","train"), os.path.join(DATA_PATH, "lolol"),)
    
