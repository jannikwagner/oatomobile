# Copyright 2020 The OATomobile Authors. All Rights Reserved.
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
"""Handles the hosted CARLA autopilot expert demonstrations dataset."""

import glob
import os
import sys
import zipfile
from typing import Any, List
from typing import Callable
from typing import Generator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wget
from absl import logging

from oatomobile.core.dataset import Dataset
from oatomobile.core.dataset import Episode
from oatomobile.torch.networks.perception import MobileNetV2
from defaults import device


class CARLADataset(Dataset):
    """The CARLA autopilot expert demonstrations dataset."""

    def __init__(
        self,
        id: str,
    ) -> None:
        """Constructs a CARLA dataset.

        Args:
          id: One of {"raw", "examples", "processed"}.
        """
        if id not in ("raw", "examples", "processed"):
            raise ValueError("Unrecognised CARLA dataset id {}".format(id))
        self.id = id
        super(CARLADataset, self).__init__()

    def _get_uuid(self) -> str:
        """Returns the universal unique identifier of the dataset."""
        return "CARLATown01Autopilot{}-v0".format(self.id)

    @property
    def info(self) -> Mapping[str, Any]:
        """The dataset description."""
        return dict(
            uuid=self.uuid,
            town="Town01",
            agent="carsuite_baselines.rulebased.Autopilot",
            noise=0.2,
        )

    @property
    def url(self) -> str:
        """The URL where the dataset is hosted."""
        return os.path.join(
            "https://www.cs.ox.ac.uk",
            "people",
            "angelos.filos",
            "data",
            "oatomobile",
            "{}.zip".format(self.id),
        )

    def download_and_prepare(self, output_dir: str) -> None:
        """Downloads and prepares the dataset from the host URL.

        Args:
          output_dir: The absolute path where the prepared dataset is stored.
        """
        # Creates the necessary output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Temporary zip file to use.
        zfname = os.path.join(output_dir, "{}.zip".format(self.id))
        # Downloads dataset from Google Drive.
        logging.debug("Starts downloading '{}' dataset".format(self.id))
        wget.download(
            url=self.url,
            out=zfname,
        )
        # Unzips data.
        logging.debug("Unzips the data from {}".format(zfname))
        with zipfile.ZipFile(zfname) as zfile:
            zfile.extractall(output_dir)
        # Removes the zip file.
        logging.debug("Removes the compressed {}".format(zfname))
        os.remove(zfname)

    @staticmethod
    def load_datum(
        fname: str,
        modalities: Sequence[str],
        mode: bool,
        dataformat: str = "HWC",
    ) -> Mapping[str, np.ndarray]:
        """Loads a single datum from the dataset.

        Args:
          fname: The absolute path to the ".npz" datum.
          modalities: The keys of the attributes to fetch.
          mode: If True, it labels its datum with {FORWARD, STOP, LEFT, RIGHT}.
          dataformat: The format of the 3D data, one of `{HWC, CHW}`.

        Returns:
          The datum in a dictionary, `NumPy`-friendly format.
        """
        assert dataformat in ("HWC", "CHW")

        dtype = np.float32
        sample = dict()

        with np.load(fname) as datum:
            for attr in modalities:
                # Fetches the value.
                sample[attr] = datum[attr]
                # Converts scalars to 1D vectors.
                sample[attr] = np.atleast_1d(sample[attr])
                # Casts value to same type.
                sample[attr] = sample[attr].astype(dtype)
                if len(sample[attr].shape) == 3 and dataformat == "CHW":
                    # Converts from HWC to CHW format.
                    sample[attr] = np.transpose(sample[attr], (2, 0, 1))

        # Appends `mode` attribute where `{0: FORWARD, 1: STOP, 2: TURN}`.
        if mode and "player_future" in sample:
            plan = sample["player_future"]
            x_T, y_T = plan[-1, :2]
            # Norm of the vector (x_T, y_T).
            norm = np.linalg.norm([x_T, y_T])
            # Angle of vector (0, 0) -> (x_T, y_T).
            theta = np.degrees(np.arccos(x_T / (norm + 1e-3)))
            if norm < 3:  # STOP
                sample["mode"] = 1
            elif theta > 15:  # LEFT
                sample["mode"] = 2
            elif theta <= -15:  # RIGHT
                sample["mode"] = 3
            else:  # FORWARD
                sample["mode"] = 0
            sample["mode"] = np.atleast_1d(sample["mode"])
            sample["mode"] = sample["mode"].astype(dtype)

        # Records the path to the sample.
        sample["name"] = fname

        return sample

    @staticmethod
    def collect(
        town: str,
        output_dir: str,
        num_vehicles: int,
        num_pedestrians: int,
        num_steps: int = 1000,
        spawn_point: Optional[Union[int, "carla.Location"]] = None,  # pylint: disable=no-member
        destination: Optional[Union[int, "carla.Location"]] = None,  # pylint: disable=no-member
        sensors: Sequence[str] = (
            "acceleration",
            "velocity",
            "lidar",
            "is_at_traffic_light",
            "traffic_light_state",
            "actors_tracker",
        ),
        render: bool = False,
        agent_fn=None,
        weather=None,
    ) -> None:
        """Collects autopilot demonstrations for a single episode on CARLA.

        Args:
          town: The CARLA town id.
          output_dir: The full path to the output directory.
          num_vehicles: The number of other vehicles in the simulation.
          num_pedestrians: The number of pedestrians in the simulation.
          num_steps: The number of steps in the simulator.
          spawn_point: The hero vehicle spawn point. If an int is
            provided then the index of the spawn point is used.
            If None, then randomly selects a spawn point every time
            from the available spawn points of each map.
          destination: The final destination. If an int is
            provided then the index of the spawn point is used.
            If None, then randomly selects a spawn point every time
            from the available spawn points of each map.
          sensors: The list of recorded sensors.
          render: If True it spawn the `PyGame` display.
        """
        from oatomobile.baselines.rulebased.autopilot.agent import AutopilotAgent
        agent_fn = AutopilotAgent if agent_fn is None else agent_fn
        from oatomobile.core.loop import EnvironmentLoop
        from oatomobile.core.rl import FiniteHorizonWrapper
        from oatomobile.core.rl import SaveToDiskWrapper
        from oatomobile.envs.carla import CARLAEnv
        from oatomobile.envs.carla import TerminateOnCollisionWrapper

        # Storage area.
        os.makedirs(output_dir, exist_ok=True)

        # Initializes a CARLA environment.
        env = CARLAEnv(
            town=town,
            sensors=sensors,
            spawn_point=spawn_point,
            destination=destination,
            num_vehicles=num_vehicles,
            num_pedestrians=num_pedestrians,
            weather=weather,
        )
        # Terminates episode if a collision occurs.
        env = TerminateOnCollisionWrapper(env)
        # Wraps the environment in an episode handler to store <observation, action> pairs.
        env = SaveToDiskWrapper(env=env, output_dir=output_dir)
        # Caps environment's duration.
        env = FiniteHorizonWrapper(env=env, max_episode_steps=num_steps)

        # Run a full episode.
        EnvironmentLoop(
            agent_fn=agent_fn,
            environment=env,
            render_mode="human" if render else "none",
        ).run()

    @staticmethod
    def process(
        dataset_dir: str,
        output_dir: str,
        future_length: int = 80,
        past_length: int = 20,
        num_frame_skips: int = 5,
        ordered=True,
        min_distance_since_last: float = 0.01,
        min_distance_trajectory: float = 0.01,
    ) -> None:
        """Converts a raw dataset to demonstrations for imitation learning.  # adds player future and player past (local locations)

        Args:
          dataset_dir: The full path to the raw dataset.
          output_dir: The full path to the output directory.
          future_length: The length of the future trajectory.
          past_length: The length of the past trajectory.
          num_frame_skips: The number of frames to skip.
          ordered: flag, if True, processed data also has order and is saved as an epsiode with metadata
          min_dist_since_last: the minim distance that needs to be moved before a new frame is accepted in the processed data
        """
        from oatomobile.utils import carla as cutil

        # Creates the necessary output directory.
        if ordered:  # episodes are used to save order in a metadata file
            if output_dir[-1] == "/":
                output_dir = output_dir[:-1]
            parent_dir, token = os.path.split(output_dir)
            output_episode = Episode(parent_dir, token)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Iterate over all episodes.
        for episode_token in tqdm.tqdm(os.listdir(dataset_dir)):
            logging.debug("Processes {} episode".format(episode_token))
            # Initializes episode handler.
            episode = Episode(parent_dir=dataset_dir, token=episode_token)
            # Fetches all `.npz` files from the raw dataset.
            try:
                sequence = episode.fetch()  # list of tokens for frames in order fetched from metadata
            except FileNotFoundError:
                continue

            # Always keep `past_length+future_length+1` files open.
            if not len(sequence) >= past_length + future_length + 1:
                continue
            old_location = None
            for i in tqdm.trange(
                past_length,
                len(sequence) - future_length,
                num_frame_skips,
            ):
                try:
                    # Player context/observation.
                    observation = episode.read_sample(sample_token=sequence[i])
                    current_location = observation["location"]
                    current_rotation = observation["rotation"]

                    if old_location is None:
                        old_location = current_location
                        distance_since_last = np.inf
                    else:
                        distance_since_last = np.sum(
                            (current_location - old_location)**2)**0.5
                    # print(distance)
                    if distance_since_last < min_distance_since_last:  # we didn't move
                        continue
                    else:
                        old_location = current_location

                    # Build past trajectory.
                    player_past = list()
                    for j in range(past_length, 0, -1):
                        past_location = episode.read_sample(
                            sample_token=sequence[i - j],
                            attr="location",
                        )
                        player_past.append(past_location)
                    player_past = np.asarray(player_past)
                    assert len(player_past.shape) == 2
                    player_past = cutil.world2local(
                        current_location=current_location,
                        current_rotation=current_rotation,
                        world_locations=player_past,
                    )

                    # Build future trajectory.
                    player_future = list()
                    for j in range(1, future_length + 1):
                        future_location = episode.read_sample(
                            sample_token=sequence[i + j],
                            attr="location",
                        )
                        player_future.append(future_location)
                    player_future = np.asarray(player_future)
                    assert len(player_future.shape) == 2
                    player_future = cutil.world2local(
                        current_location=current_location,
                        current_rotation=current_rotation,
                        world_locations=player_future,
                    )
                    distance_trajetory = np.sum(
                        (player_future[-1] - player_past[0])**2)**0.5
                    # print(distance_trajetory)
                    if distance_trajetory < min_distance_trajectory:
                        continue

                    # Store to ouput directory.
                    if ordered:
                        output_episode.append(**observation,
                                              player_future=player_future,
                                              player_past=player_past,)
                    else:
                        np.savez_compressed(
                            os.path.join(
                                output_dir, "{}.npz".format(sequence[i])),
                            **observation,
                            player_future=player_future,
                            player_past=player_past,
                        )

                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        sys.exit(0)

    @staticmethod
    def plot_datum(
        fname: str,
        output_dir: str,
    ) -> None:
        """Visualizes a datum from the dataset.

        Args:
          fname: The absolute path to the datum.
          output_dir: The full path to the output directory.
        """
        from oatomobile.utils import graphics as gutil

        COLORS = [
            "#0071bc",
            "#d85218",
            "#ecb01f",
            "#7d2e8d",
            "#76ab2f",
            "#4cbded",
            "#a1132e",
        ]

        # Creates the necessary output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Load datum.
        datum = np.load(fname)

        # Draws LIDAR.
        if "lidar" in datum:
            bev_meters = 25.0
            lidar = gutil.lidar_2darray_to_rgb(datum["lidar"])
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(
                np.transpose(lidar, (1, 0, 2)),
                extent=(-bev_meters, bev_meters, bev_meters, -bev_meters),
            )
            ax.set(frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.savefig(
                os.path.join(output_dir, "lidar.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

        # Draws first person camera-view.
        if "front_camera_rgb" in datum:
            front_camera_rgb = datum["front_camera_rgb"]
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(front_camera_rgb)
            ax.set(frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.savefig(
                os.path.join(output_dir, "front_camera_rgb.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

        # Draws bird-view camera.
        if "bird_view_camera_cityscapes" in datum:
            bev_meters = 25.0
            bird_view_camera_cityscapes = datum["bird_view_camera_cityscapes"]
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(
                bird_view_camera_cityscapes,
                extent=(-bev_meters, bev_meters, bev_meters, -bev_meters),
            )
            # Draw past if available.
            if "player_past" in datum:
                player_past = datum["player_past"]
                ax.plot(
                    player_past[..., 1],
                    -player_past[..., 0],
                    marker="x",
                    markersize=4,
                    color=COLORS[0],
                    alpha=0.15,
                )
            # Draws future if available.
            if "player_future" in datum:
                player_future = datum["player_future"]
                ax.plot(
                    player_future[..., 1],
                    -player_future[..., 0],
                    marker="o",
                    markersize=4,
                    color=COLORS[1],
                    alpha=0.15,
                )
            # Draws goals if available.
            if "goal" in datum:
                goal = datum["goal"]
                ax.plot(
                    goal[..., 1],
                    -goal[..., 0],
                    marker="D",
                    markersize=6,
                    color=COLORS[2],
                    linestyle="None",
                    alpha=0.25,
                    label=r"$\mathcal{G}$",
                )
            ax.set(frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.savefig(
                os.path.join(output_dir, "bird_view_camera_cityscapes.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

        # Draws bird-view camera.
        if "bird_view_camera_rgb" in datum:
            bev_meters = 25.0
            bird_view_camera_rgb = datum["bird_view_camera_rgb"]
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(
                bird_view_camera_rgb,
                extent=(-bev_meters, bev_meters, bev_meters, -bev_meters),
            )
            # Draw past if available.
            if "player_past" in datum:
                player_past = datum["player_past"]
                ax.plot(
                    player_past[..., 1],
                    -player_past[..., 0],
                    marker="x",
                    markersize=4,
                    color=COLORS[0],
                    alpha=0.15,
                )
            # Draws future if available.
            if "player_future" in datum:
                player_future = datum["player_future"]
                ax.plot(
                    player_future[..., 1],
                    -player_future[..., 0],
                    marker="o",
                    markersize=4,
                    color=COLORS[1],
                    alpha=0.15,
                )
            ax.set(frame_on=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.savefig(
                os.path.join(output_dir, "bird_view_camera_rgb.png"),
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

    @classmethod
    def plot_coverage(
        cls,
        dataset_dir: str,
        output_fname: str,
        color: int = 0,
    ) -> None:
        """Visualizes all the trajectories in the dataset.

        Args:
          dataset_dir: The parent directory of all the dataset.
          output_fname: The full path to the output filename.
          color: The index of the color to use for the trajectories.
        """
        COLORS = [
            "#0071bc",
            "#d85218",
            "#ecb01f",
            "#7d2e8d",
            "#76ab2f",
            "#4cbded",
            "#a1132e",
        ]

        # Fetches all the data points.
        data_files = glob.glob(
            os.path.join(dataset_dir, "**", "*.npz"),
            recursive=True,
        )

        # Container that stores all locaitons.
        locations = list()
        for npz_fname in tqdm.tqdm(data_files):
            try:
                locations.append(
                    cls.load_datum(
                        npz_fname,
                        modalities=["location"],
                        mode=False,
                    )["location"])
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit(0)
        locations = np.asarray(locations)

        # Scatter plots all locaitons.
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.scatter(
            locations[..., 0],
            locations[..., 1],
            s=5,
            alpha=0.01,
            color=COLORS[color % len(COLORS)],
        )
        ax.set(title=dataset_dir, frame_on=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(
            os.path.join(output_fname),
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

    @classmethod
    def as_tensorflow(
        cls,
        dataset_dir: str,
        modalities: Sequence[str],
        mode: bool = False,
    ) -> "tensorflow.data.Dataset":
        """Implements a data reader and loader for the expert demonstrations.

        Args:
          dataset_dir: The absolute path to the raw dataset.
          modalities: The keys of the attributes to fetch.
          mode: If True, it labels its datum with {FORWARD, STOP, LEFT, RIGHT}.

        Returns:
          The unbatched `TensorFlow` dataset.
        """
        import tensorflow as tf

        # Fetches all the filenames.
        filenames = glob.glob(os.path.join(dataset_dir, "*.npz"))

        # Gets shapes of output tensors.
        output_shapes = dict()
        with np.load(filenames[0]) as datum:
            for modality in modalities:
                output_shapes[modality] = tf.TensorShape(
                    np.atleast_1d(datum[modality]).shape)

        # Appends "mode" attribute.
        if mode:
            output_shapes["mode"] = tf.TensorShape((1,))

        # Sets all output types to `tf.float32`.
        output_types = {
            modality: tf.float32 for modality in output_shapes.keys()}

        return tf.data.Dataset.from_generator(
            generator=lambda: (cls.load_datum(
                npz_fname,
                modalities,
                mode,
                dataformat="HWC",
            ) for npz_fname in filenames),
            output_types=output_types,
            output_shapes=output_shapes,
        )

    @classmethod
    def as_numpy(
        cls,
        dataset_dir: str,
        modalities: Sequence[str],
        mode: bool = False,
    ) -> Generator[Mapping[str, np.ndarray], None, None]:
        """Implements a data reader and loader for the expert demonstrations.

        Args:
          dataset_dir: The absolute path to the raw dataset.
          modalities: The keys of the attributes to fetch.
          mode: If True, it labels its datum with {FORWARD, STOP, LEFT, RIGHT}.

        Returns:
          The unbatched `NumPy` dataset.
        """
        import tensorflow_datasets as tfds

        return tfds.as_numpy(cls.as_tensorflow(dataset_dir, modalities, mode))

    @classmethod
    def as_torch(
        cls,
        dataset_dir: str,
        modalities: Sequence[str],
        transform: Optional[Callable[[Any], Any]] = None,
        mode: bool = False,
        only_array: bool = False,
    ) -> "torch.utils.data.Dataset":
        """Implements a data reader and loader for the expert demonstrations.

        Args:
          dataset_dir: The absolute path to the raw dataset.
          modalities: The keys of the attributes to fetch.
          transform: The transformations applied on each datum.
          mode: If True, it labels its datum with {FORWARD, STOP, LEFT, RIGHT}.
          only_array: If True, it removes all the keys that are non-array, useful
            when training a model and want to run `.to(device)` without errors.

        Returns:
          The unbatched `PyTorch` dataset.
        """
        import torch

        class PyTorchDataset(torch.utils.data.Dataset):
            """Implementa a data reader for the expert demonstrations."""

            def __init__(
                self,
                dataset_dir: str,
                modalities: Sequence[str],
                transform: Optional[Callable[[Any], Any]] = None,
                mode: bool = False,
            ) -> None:
                """A simple `PyTorch` dataset.

                Args:
                  dataset_dir: The absolute path to the raw dataset.
                  modalities: The keys of the attributes to fetch.
                  mode: If True, it labels its datum with {FORWARD, STOP, LEFT, RIGHT}.
                """
                # Internalise hyperparameters.
                self._modalities = modalities
                self._npz_files = get_npz_files(dataset_dir)
                self._transform = transform
                self._mode = mode

            def __len__(self) -> int:
                """Returns the size of the dataset."""
                return len(self._npz_files)

            def __getitem__(
                self,
                idx: int,
            ) -> Mapping[str, np.ndarray]:
                """Loads a single datum.

                Returns:
                  The datum in `NumPy`-friendly format.
                """
                # Loads datum from dataset.
                sample = cls.load_datum(
                    fname=self._npz_files[idx],
                    modalities=self._modalities,
                    mode=self._mode,
                    dataformat="CHW",
                )

                # Filters out non-array keys.
                for key in list(sample):
                    if not isinstance(sample[key], np.ndarray):
                        sample.pop(key)

                # Applies (optional) transformation to all values.
                if self._transform is not None:
                    sample = {key: self._transform(val)
                              for (key, val) in sample.items()}
                return sample

        return PyTorchDataset(dataset_dir, modalities, transform, mode)

    @classmethod
    def annotate_with_model(cls,
                            dataset_dir: str,
                            modalities: Sequence[str],
                            model: MobileNetV2,
                            model_name: str,
                            transform: Optional[Callable[[Any], Any]] = None,
                            mode: bool = False,
                            num_instances: int = None,
                            device="cpu",
                            recursive=True):
        """add the ouput of a mobilenet to each frame of a dataset

        Args:
            dataset_dir (str): the root path of the dataset
            modalities (Sequence[str]): the features of the data the models needs to see as inputs
            model (MobileNetV2): the model whose outputs are added to the data
            model_name (str): the name of the feature of the models outputs (should be name of the model)
            transform (Optional[Callable[[Any], Any]], optional): optional transform of the data. Defaults to None.
            mode (bool, optional): see load_datum. Defaults to False.
            num_instances (int, optional): the number of frames that should be annotated (for debugging). Defaults to None.
            device (str, optional): the device on which the model is stored. Defaults to "cpu".
            recursive (bool, optional): whether the dataset should be loaded recursively (subdirectories as well). Defaults to True.
        """
        from oatomobile.torch import transforms
        import torch

        npz_files = get_npz_files(dataset_dir, recursive)
        for npz_file in tqdm.tqdm(npz_files[:num_instances]):
            # prepare sample
            sample = cls.load_datum(
                fname=npz_file,
                modalities=modalities,
                mode=mode,
                dataformat="CHW",
            )

            # Filters out non-array keys.
            for key in list(sample):
                if not isinstance(sample[key], np.ndarray):
                    sample.pop(key)

            # Applies (optional) transformation to all values.
            if transform is not None:
                sample = {key: transform(val) for (key, val) in sample.items()}

            if "lidar" in sample:
                sample["visual_features"] = sample.pop("lidar")

            # Preprocesses the visual features.
            if "visual_features" in sample:
                lidar = torch.from_numpy(sample["visual_features"])
                lidar = lidar.to(device=device)
                lidar = lidar.view(1, *lidar.size())
                lidar = transforms.transpose_visual_features(
                    transforms.downsample_visual_features(
                        visual_features=lidar,
                        output_shape=(100, 100),
                    ))
            lidar = lidar.to(device)
            model_output = model(lidar)[0].cpu()

            with np.load(
                npz_file,
                allow_pickle=True,
            ) as npz:
                observation = dict()
                for _attr in npz:
                    observation[_attr] = npz[_attr]
            observation[model_name] = model_output.detach().cpu().numpy()

            np.savez_compressed(npz_file,
                                **observation,
                                )

    @classmethod
    def make_arff(cls,
                  dataset_dir: str,
                  outpath: str,
                  modalities: Sequence[str],
                  relation_name: str,
                  comments: Optional[List[str]] = None,
                  mode: bool = False,
                  recursive=True,
                  dataformat="HWC",
                  num_timesteps_to_keep: int = 4,
                  num_instances=None):
        """create an arff file from a dataset. Different episodes are appended lexicograhpically.
        Different frames in an episode are appended in their order if the episode is ordered and lexicographically otherwise.

        Args:
            dataset_dir (str): the path of the dataset
            outpath (str): the path of the to be saved arff file
            modalities (Sequence[str]): a tuple of the features that should be stored in the arff file.
              Multidimensional features (such as lidar sensor data) will be multiple columns in the arff file, one for each dimension.
            relation_name (str): the name of the relation that will be set in the arff file
            comments (Optional[List[str]], optional): optional comments in the header of the arff file. Defaults to None.
            mode (bool, optional): If True, datums are labeled with {FORWARD, STOP, LEFT, RIGHT} (See load_datum). Defaults to False.
            recursive (bool, optional): whether subdirectories of the data path should be considered.
              Yes for data with multiple distributions. Defaults to True.
            dataformat (str, optional): The format of the 3D data, one of `{HWC, CHW}`. Defaults to "HWC".
            num_timesteps_to_keep (int, optional): The number of timesteps of the trajectory. Defaults to 4.
            num_instances ([type], optional): the maximum number of instances in the arff file. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        npz_files = get_npz_files(dataset_dir, recursive)
        with open(outpath, "w") as arff_file:
            if comments is not None:
                for comment in comments:
                    arff_file.write("% {}\n".format(comment))

            arff_file.write("@RELATION " + relation_name + "\n")

            for i, npz_file in enumerate(tqdm.tqdm(npz_files[:num_instances])):
                observation = cls.load_datum(
                    npz_file, modalities, mode, dataformat)
                # transform (maybe use extra function)
                if "player_future" in observation:
                    T, _ = observation["player_future"].shape
                    increments = T // num_timesteps_to_keep
                    observation["player_future"] = observation["player_future"][0::increments, :]

                if i == 0:  # first iteration -> add metadata about attributes
                    for key in modalities:
                        value = observation[key]
                        if isinstance(value, np.ndarray):
                            for i in range(len(value.flat)):
                                arff_file.write(
                                    "@ATTRIBUTE {}_{} NUMERIC\n".format(key, i))
                        elif isinstance(value, int) or isinstance(value, float):
                            arff_file.write(
                                "@ATTRIBUTE {} NUMERIC\n".format(key))
                        else:
                            raise NotImplementedError(key, value)
                    arff_file.write("\n@DATA\n")

                line = get_observation_line(observation, modalities) + "\n"
                arff_file.write(line)

    @classmethod
    def car_not_moving_counts(cls, episode_path: str, eps: float = 0.01) -> List[int]:
        """Count how many frames in a row the car is not moving in an episode.

        Args:
            episode_path (str): the episode path
            eps (float, optional): the distance threshold. If the distance is higher, the car is considered to be moving.
              0.01 seems to be a good threshold since standing cars often seem to move around 0.0005. Defaults to 0.01.

        Returns:
            List[int]: an entry means that a car hasn't moved for that many frames. 0 means the car is moving.
              The sum is the total number of frames a car hasn't moved.
        """
        if episode_path[-1] == "/":
            episode_path = episode_path[:-1]
        episode_path, token = os.path.split(episode_path)

        episode = Episode(parent_dir=episode_path, token=token)
        # Fetches all `.npz` files from the raw dataset.
        try:
            sequence = episode.fetch()  # list of tokens for frames in order fetched from metadata
        except FileNotFoundError:
            raise

        old_location = 0
        old_rotation = None
        counter = 0
        counts = [0]
        for i in tqdm.trange(
            len(sequence)
        ):
            try:
                # Player context/observation.
                observation = episode.read_sample(sample_token=sequence[i])
                current_location = observation["location"]
                current_rotation = observation["rotation"]
                if old_location is None:
                    old_location, old_rotation = current_location, current_rotation
                distance = np.sum((current_location - old_location)**2)**0.5
                # print(distance)
                if distance < eps:  # and old_rotation == current_rotation:
                    counter += 1
                else:
                    counts.append(counter)
                    counter = 0
                    old_location = current_location
                    old_rotation = current_rotation
                # print(i,":",counter)
            except:
                print("Skipped", i)
        if counter != 0:
            counts.append(counter)
        return counts


def get_observation_line(observation: Mapping[str, np.ndarray], modalities: Sequence[str], round=10) -> str:
    """Create one line of an arff file for one observation/fram of a dataset.
    Multidimensional features will be split into a column for each dimension.

    Args:
        observation (Mapping[str, np.ndarray]): a frame of a dataset, transformed into an arff line
        modalities (Sequence[str]): the features
        round (int, optional): the number of decimals in float features. Defaults to 10.

    Raises:
        NotImplementedError: if a feature is of an unknown type

    Returns:
        str: the arff line representing the frame
    """
    values = []
    for key in modalities:
        value = observation[key]
        if isinstance(value, np.ndarray):
            if round is not False:
                value = np.round(value, round)
            for i in range(len(value.flat)):
                val = str(value.flat[i])
                values.append(val)

        elif isinstance(value, int) or isinstance(value, float):
            if round is not False:
                value = str(np.round(value, round))
                values.append(value)

        else:
            raise NotImplementedError(key, value)

        return ",".join(values)


def get_npz_files(dataset_dir: str, recursive=True) -> List[str]:
    """Get a list of the absolute path of all npz files (frames) in a data set.
    If the dataset is ordered, the list is in that order.

    Args:
        dataset_dir (str): the root path of the dataset
        recursive (bool, optional): whether subdirectories should be recursively considered
          (necessary for datasets consisting of several distributions). Defaults to True.

    Returns:
        List[str]: the list of absolute paths of npz files
    """
    local_listdir = os.listdir(dataset_dir)
    global_listdir = [os.path.join(dataset_dir, x) for x in local_listdir]
    npz_files = []
    if recursive:
        subdirs = [x for x in sorted(global_listdir) if os.path.isdir(x)]
        datasets = [get_npz_files(x) for x in subdirs]
        for subdir_files in datasets:
            npz_files.extend(subdir_files)

    if "metadata" in local_listdir:  # ordered dataset
        with open(os.path.join(dataset_dir, "metadata")) as metadata:
            samples = metadata.read()
        samples = list(filter(None, samples.split("\n")))
        npz_files.extend([os.path.join(dataset_dir, token+".npz")
                         for token in samples])

    else:  # unordered dataset (original)
        npz_files.extend(glob.glob(os.path.join(
            dataset_dir, "*.npz"), recursive=False))

    return npz_files
