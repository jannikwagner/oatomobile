# Copyright 2021 Jannik Wagner. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This file offers functions to conduct specific data generation and manipulation tasks with oatomobile
"""
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
from typing import Tuple, Callable

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

WEATHERS = {  # all weathers supported by carla
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "Default": carla.WeatherParameters.Default,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset
}

ALL_SENSORS = (  # all sensors of OATomobile environment on carla
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


def download():
    """ download the data from oatomobile
    """
    raw = CARLADataset("raw")
    examples = CARLADataset("examples")
    processed = CARLADataset("processed")
    datasets = [raw, examples, processed]
    for dataset in datasets:
        dataset.download_and_prepare(os.path.join(DATA_PATH, "downloaded"))
        

def visualize_raw_rgb(episode_path: str, outpath: str,
                      sensors: Tuple[str] = ("front_camera_rgb",
                                             "rear_camera_rgb",
                                             "left_camera_rgb",
                                             "right_camera_rgb",
                                             "bird_view_camera_rgb",
                                             "bird_view_camera_cityscapes",
                                             "lidar"),
                      start=None, end=None, step=None):
    """extract visual sensor data from frames of an episode and store as image data

    Args:
        episode_path (str): the path of the episode
        outpath (str): the path where the images should be saved, will contain a folder for each sensor
        sensors (Tuple[str], optional): The sensors which should be used. If a sensor is not contained in the data the folder is left empty. Defaults to ( "front_camera_rgb", "rear_camera_rgb", "left_camera_rgb", "right_camera_rgb", "bird_view_camera_rgb", "bird_view_camera_cityscapes", "lidar").
        start ([type], optional): start fram
        end ([type], optional): end fram
        step ([type], optional): stepsize of frames
    """
    output_dirs = {sensor: os.path.join(outpath, sensor) for sensor in sensors}
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    episode = Episode(episode_path, "")
    for i, line in enumerate(tqdm.tqdm(episode.fetch()[start:end:step])):
        x = episode.read_sample(line, None)
        for sensor, output_dir in output_dirs.items():
            if sensor in x:
                img = x[sensor]
                if sensor == "lidar":
                    img2 = np.zeros(img.shape[:2]+(3,))
                    img2[:, :, :2] = img
                    img = img2
                    pass
                plt.imsave(os.path.join(
                    output_dir, "{a}_{i}.png".format(a=sensor, i=i)), img)


def imgs_to_gif(inpath: str, outfile: str, prefix: str, start: int = 0, end: int = None):
    """Take all images in a path with name prefix_k, for k=0,... and concatenate them to a gif

    Args:
        inpath (str): the path of the folder containing the images
        outfile (str): the path of the target gif
        prefix (str): only files with this prefix are used
        start (int, optional): the start index. Defaults to 0.
        end (int, optional): the end index. If None, uses all files. Defaults to None.
    """
    count = len(os.listdir(inpath))
    if end is None:
        end = count
    assert start <= end <= count
    images = []
    for i in tqdm.trange(start, end):
        filename = os.path.join(
            inpath, "{prefix}_{i}.png".format(prefix=prefix, i=i))
        images.append(imageio.imread(filename))
    imageio.mimsave(outfile, images)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "lidar"), os.path.join(DATA_PATH, "visualization", "rgb", "lidar.gif"), "c", 100, 200)
#    imgs_to_gif(os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb"), os.path.join(DATA_PATH, "visualization", "rgb", "front_camera_rgb.gif"), "front_camera_rgb", 100, 400)


def generate_distributions(root_path: str, sensors: Tuple[str] = ("acceleration",
                                                                  "velocity",
                                                                  "lidar",
                                                                  "is_at_traffic_light",
                                                                  "traffic_light_state",
                                                                  "actors_tracker",),
                           n_frames: int = 2000, n_episodes: int = 20, weathers: Tuple[str] = ("HardRainNoon", "ClearNoon"),
                           n_ped_cars: int = (0, 50), towns: Tuple[str] = ("Town01", "Town03")):
    """generate episodes for each combination of town, weather and traffic intensity

    Args:
        root_path (str): Output path for data generation. For each distribution a folder will be saved.
        sensors (Tuple[str], optional): The sensors that will be retrieved for each fram. Defaults to ("acceleration", "velocity", "lidar", "is_at_traffic_light", "traffic_light_state", "actors_tracker",).
        n_frames (int, optional): The number of frames for each episode. Defaults to 2000.
        n_episodes (int, optional): The number of episodes for each distribution. Defaults to 20.
        weathers (Tuple[str], optional): The different weather options. Defaults to ("HardRainNoon", "ClearNoon").
        n_ped_cars (int, optional): The different traffic intensities. Defaults to (0, 50).
        towns (Tuple[str], optional): The different towns. Defaults to ("Town01", "Town03").
    """
    agent_fn = AutopilotAgent
    for weather, n, town, i in tqdm.tqdm(list(itertools.product(weathers, n_ped_cars, towns, range(n_episodes)))):
        path = os.path.join(root_path, town+weather+str(n))
        collect_not_moving_counts(
            town, path, n, n, n_frames, sensors, agent_fn, weather)


def collect_not_moving_counts(town: str, output_dir: str, num_vehicles: int, num_pedestrains: int, n_frames: int,
                              sensors: Tuple[str], agent_fn: Callable, weather: str, visualize: bool = True, max_not_moving=0.5,
                              max_not_moving_end: float = 0.4):
    """collects an episode and makes sure the car is moving most of the time

    Args:
        town (str): the town in which the car drives
        output_dir (str): path which should contain the episode folder
        num_vehicles (int): the number of vehicles spawned
        num_pedestrains (int): the number of pedestrains spawned
        n_frames (int): the number of frames of the episode
        sensors (Tuple[str]): the sensors whose data should be retrieved from carla and stored
        agent_fn (Callable): the agent which controls the car
        weather (str): the weather of the simulation (see WEATHERS)
        visualize (bool, optional): whether the image sensor data should be saved as images. Defaults to True.
        max_not_moving (float, optional): the maximum proportion of frames where the car is not moving. Defaults to 0.5.
        max_not_moving_end (float, optional): the maximum proportion of ending frames where the car is not moving. Defaults to 0.4.
    """
    while True:
        os.makedirs(output_dir, exist_ok=True)
        listdir = os.listdir(output_dir)
        CARLADataset.collect(town, output_dir, num_vehicles, num_pedestrains,
                             n_frames, None, None, sensors, False, agent_fn, weather)
        newdir = [x for x in os.listdir(
            output_dir) if x not in listdir][0]  # find new folder
        newdir_path = os.path.join(output_dir, newdir)
        counts = CARLADataset.car_not_moving_counts(newdir_path)
        print(counts)
        if sum(counts) < max_not_moving*n_frames and counts[-1] < max_not_moving_end*n_frames:
            break
        shutil.rmtree(newdir_path)
        print("repeat", weather, num_vehicles, town)

    if visualize:
        vis_path = os.path.join(output_dir, "vis", newdir)
        os.makedirs(vis_path, exist_ok=True)
        with open(os.path.join(vis_path, "not_moving_counts.txt"), "w") as counts_file:
            counts_file.write(str(counts))
        visualize_raw_rgb(episode_path=newdir_path, outpath=vis_path,)


def process_distributions(inpath: str, outpath: str, num_frame_skips: int = 5,
                          min_distance_since_last: float = 0.01, min_distance_trajectory: float = 0.01):
    """Process all episodes of all distributions 

    Args:
        inpath (str): the path containing the different episodes
        outpath (str): the path that will contain the processed dataset
        num_frame_skips (int, optional): every so many frames are used. Defaults to 5.
        min_distance_since_last (float, optional): the minimum distance since the last used frame so that a frame is used. Defaults to 0.01.
        min_distance_trajectory (float, optional): the minimum distance to the end of the future trajectory so that the fram is used. Defaults to 0.01.
    """
    for dist in os.listdir(inpath):
        CARLADataset.process(os.path.join(inpath, dist), os.path.join(outpath, dist), num_frame_skips=num_frame_skips,
                             min_distance_since_last=min_distance_since_last, min_distance_trajectory=min_distance_trajectory)


if __name__ == "__main__":
    # it would be beneficial to create a management class similar to a jobcenter
    if False:  # generate a dataset for training a dim consisting of several episodes of several target distributions
        root_path = os.path.join(DATA_PATH, "mydistributions", "raw", "train",)
        generate_distributions(root_path,  sensors=("acceleration",
                                                    "velocity",
                                                    "lidar",
                                                    "is_at_traffic_light",
                                                    "traffic_light_state",
                                                    "actors_tracker",),
                               n_frames=2000, n_episodes=20,
                               # the different weather options
                               weathers=("HardRainNoon", "ClearNoon"),
                               # the different traffic intensities
                               n_ped_cars=(0, 50),
                               towns=("Town01", "Town03"))  # the different towns

    if False:  # process a dataset for training containing episodes of different target distributions
        raw_path = os.path.join(DATA_PATH, "mydistributions", "raw", "train",)
        processed_path = os.path.join(
            DATA_PATH, "mydistributions", "processed", "train")
        process_distributions(raw_path, processed_path, num_frame_skips=5,
                              min_distance_since_last=0, min_distance_trajectory=0)

    if False:  # generate episodes for a data stream for the drift detection
        root_path = os.path.join(DATA_PATH, "mystream", "raw", "test")
        d0 = ["Town01", "ClearNoon", 0]
        d1 = ["Town03", "HardRainNoon", 50]
        dist_list = [d0, d1]
        order = [d0, d0, d1, d1, d0, d0, d1, d1]  # the target stream section
        n_frames = 2000
        os.makedirs(root_path, exist_ok=True)
        target = list(np.asarray(
            [[dist_list.index(d)]*n_frames for d in order]).flat)
        with open(os.path.join(root_path, "target_dists.txt"), "w") as f:
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
        for i, (town, weather, n) in enumerate(order):
            path = os.path.join(root_path, str(i)+"_"+town+weather+str(n))
            collect_not_moving_counts(
                town, path, n, n, n_frames, sensors, agent_fn, weather)

    if False:  # create arff for datastream from episodes
        ckpt_path = os.path.join(
            MODELS_PATH, "dim", "downloaded_d128", "ckpts", "model-108.pt")
        data_path = os.path.join(DATA_PATH, "mystream", "raw", "test")
        arff_path = os.path.join(
            DATA_PATH, "mystream", "raw", "dists8_mobilenet_downloaded_d128.arff")
        model = getDIM(ckpt_path, device, 128)
        mobilenet = dict(model.named_children())["_encoder"]
        modalities = (
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "velocity",
        )
        CARLADataset.annotate_with_model(
            data_path, modalities, mobilenet, "mobilenet_downloaded_d128_e108", device=device)
        CARLADataset.make_arff(data_path, arff_path, (
            "mobilenet_downloaded_d128_e108",), "mystream_mobilenet_downloaded_d128_e108")

    if False:  # visualize a dataset containing episodes different target distributions
        root_path = os.path.join(DATA_PATH, "mydistributions", "raw", "test")
        root_outpath = os.path.join(
            DATA_PATH, "mydistributions", "raw", "test_vis")
        for dist in os.listdir(root_path):
            dist_path = os.path.join(root_path, dist)
            for episode in os.listdir(dist_path):
                episode_path = os.path.join(dist_path, episode)
                outpath = os.path.join(root_outpath, dist, episode)
                visualize_raw_rgb(episode_path=episode_path, outpath=outpath)

    if False:  # get gifs
        sensor = "lidar"
        root_path = os.path.join(DATA_PATH, "mydistributions", "raw", "test")
        for dist in os.listdir(root_path):
            dist_path = os.path.join(root_path, dist, "vis")
            if os.path.isdir(dist_path):
                episode = os.listdir(dist_path)[0]
                episode_path = os.path.join(dist_path, episode)
                rgb_path = os.path.join(episode_path, sensor)
                gif_path = os.path.join(root_path, dist+"_"+sensor+".gif")
                imgs_to_gif(rgb_path, gif_path, sensor, start=1300, end=1400)

    if False:  # let a DIM Agent control a car.
        n_frames = 2000

        ckpt_path = os.path.join(
            MODELS_PATH, "dim", "dists7.2_d32", "ckpts", "model-200.pt")
        dim = getDIM(ckpt_path, device=device, mobilenet_num_classes=32)
        agent_fn = get_agent_fn(dim)
        data_path = os.path.join(DATA_PATH, "test_dim3", "dists7.2_moving_d32")
        collect_not_moving_counts(
            "Town01", data_path, 50, 50, n_frames, ALL_SENSORS, agent_fn, "ClearNoon", True, 1, 1)
