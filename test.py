import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import imageio
import glob
os.environ["CARLA_ROOT"]="/home/jannik_wagner/carla/"
#os.environ["CARLA_ROOT"]="/home/jannik/carla/"

import oatomobile
import oatomobile.envs
from oatomobile.datasets.carla import CARLADataset
import oatomobile.baselines.torch.dim.train as train
from oatomobile.core.dataset import Episode


PATH = os.path.join(os.getcwd())
DATA_PATH = os.path.join(PATH, "data")
MODELS_PATH = os.path.join(PATH, "models")


def fun():
    sensors = (
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
    CARLADataset.collect("Town01", os.path.join(DATA_PATH, "rgb"), 100, 100, 1000, None, None, sensors)


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


def download():
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


def visualize_raw_lidar(path=None):
    if path is None:
        path = os.path.join(DATA_PATH, "rgb")
        path = os.path.join(path, random.choice(os.listdir(path)))

    output_dir1 = os.path.join(DATA_PATH, "visualization", "rgb", "lidar1")
    output_dir2 = os.path.join(DATA_PATH, "visualization", "rgb", "lidar2")
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    output_dir = os.path.join(DATA_PATH, "visualization", "rgb", "lidar")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(path, "metadata")) as metadata:
        for i, line in enumerate(tqdm.tqdm(metadata.readlines())):
            line = line[:-1] + ".npz"
            x = np.load(os.path.join(path, line))
            lidar = x.f.lidar
            plt.imsave(os.path.join(output_dir1, "a{i}.png".format(i=i)), lidar[:,:,0], cmap="gray")
            plt.imsave(os.path.join(output_dir2, "b{i}.png".format(i=i)), lidar[:,:,1], cmap="gray")
            img = np.zeros(lidar.shape[:2]+(3,))
            img[:,:,:2] = lidar
            plt.imsave(os.path.join(output_dir, "c{i}.png".format(i=i)), img)


def visualize_raw_rgb(sensors=(
        "front_camera_rgb",
        "rear_camera_rgb",
        "left_camera_rgb",
        "right_camera_rgb",
        "bird_view_camera_rgb",
        "bird_view_camera_cityscapes"), path=os.path.join(DATA_PATH, "rgb"), token=None):
    if token is None:
        token = random.choice(os.listdir(path))

    output_dirs = {sensor: os.path.join(DATA_PATH, "visualization", "rgb", sensor) for sensor in sensors}
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    episode = Episode(path, token)
    for i, line in enumerate(tqdm.tqdm(episode.fetch())):
        x = episode.read_sample(line, None)
        for sensor, output_dir in output_dirs.items():
            if sensor in x:
                plt.imsave(os.path.join(output_dir, "{a}{i}.png".format(a=sensor,i=i)), x[sensor])



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


if __name__=="__main__":
    pass
