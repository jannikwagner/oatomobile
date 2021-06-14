"""This file contains scripts to evaluate a DIM model with evaluation measures such as Average displacement error.
See main below.
"""

from defaults import PATH, MODELS_PATH, DATA_PATH, device

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
from typing import List, Mapping
import pandas as pd


def getDIM(path: str = None, device: str = "cpu", mobilenet_num_classes: int = 128) -> ImitativeModel:
    """load a DIM model from a checkpint.

    Args:
        path (str, optional): The path of the checkpoint. If None, no checkpoint is loaded. Defaults to None.
        device (str, optional): the device on which the model should be stored. cuda or cpu. Defaults to "cpu".
        mobilenet_num_classes (int, optional): The output dimension of the mobilenet. Defaults to 128.

    Returns:
        ImitativeModel: the loaded DIM
    """
    model = ImitativeModel(mobilenet_num_classes=mobilenet_num_classes)
    if path is not None:
        x = torch.load(path)
        model.load_state_dict(x)
    model.eval().to(device)
    return model


def get_agent_fn(model: ImitativeModel) -> Callable[oatomobile.Env, DIMAgent]:
    """get a function that creates a DIMAgent from a given model

    Args:
        model (ImitativeModel): the DIM model

    Returns:
        Callable[oatomobile.Env, DIMAgent]: the function which creates an agent in a given environment
    """
    def agent_fn(environment: oatomobile.Env) -> DIMAgent:
        return DIMAgent(environment, model=model)
    return agent_fn


def transform(model: ImitativeModel, batch: Mapping[str, torch.Tensor],
              device: str) -> Mapping[str, torch.Tensor]:
    """Preprocesses a batch for the model.

    Args:
        batch: (keyword arguments) The raw batch variables.

    Returns:
        The processed batch.
    """
    # Sends tensors to `device`.
    batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
    # Preprocesses batch for the model.
    batch = model.transform(batch)
    return batch


def ADE(pred: torch.Tensor, ground_truth: torch.Tensor):
    return (pred[:, :, :2] - ground_truth[:, :, :2]).square().sum(axis=-1).sqrt().mean(axis=-1)


def FDE(pred: torch.Tensor, ground_truth: torch.Tensor):
    return (pred[:, -1, :2] - ground_truth[:, -1, :2]).square().sum(axis=-1).sqrt()


def minADE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_ADE = [ADE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_ADE).min(axis=0)[0]


def minFDE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_FDE = [FDE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_FDE).min(axis=0)[0]


def evaluate(ckpt_path: str, data_path: str, output_path: str, mobilenet_num_classes: int = 128, num_batches: int = np.inf):
    """Calculate evaluation measures of a DIM model over a test data set and save it in a csv file.

    Args:
        ckpt_path (str): the path of the model checkpoint
        data_path (str): thepath of the test data used for the evaluation
        output_path (str): where the csv will be saved
        mobilenet_num_classes (int, optional): the output dimension of the mobilnet of the DIM. Defaults to 128.
        num_batches (int, optional): The number of batches of the test data over which the measures are averaged. Defaults to np.inf.
    """
    model = getDIM(ckpt_path, device, mobilenet_num_classes)
    modalities = (
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "player_future",
        "velocity",
    )
    batch_size = 128
    dataloader = CARLADataset.as_torch(data_path, modalities)

    dataset = CARLADataset.as_torch(
        dataset_dir=data_path,
        modalities=modalities,
    )
    dataloader = torch.utils.data.DataLoader(  # errors with num_workers > 1
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    k = 5
    ADE_ms = []
    ADE_10s = []  # the number of steps
    ADE_100s = []
    minADE_ks = []
    FDE_ms = []
    FDE_10s = []
    FDE_100s = []
    minFDE_ks = []
    torch.cuda.empty_cache()
    with tqdm.tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):  # gives errors with num_workers > 1
            # if True:
            # Prepares the batch.

            batch = transform(model, batch, device)
            ground_truth = batch["player_future"]
            y_samples = [model.sample(**batch).detach() for _ in range(k)]
            minADE_ks.append(minADE(y_samples, ground_truth))
            minFDE_ks.append(minFDE(y_samples, ground_truth))
            del y_samples
            y_mean = model.mean(**batch).detach()
            ADE_ms.append(ADE(y_mean, ground_truth))
            FDE_ms.append(FDE(y_mean, ground_truth))
            del y_mean
            y_10 = model.forward(10, **batch).detach()
            y_100 = model.forward(100, **batch).detach()
            ADE_10s.append(ADE(y_10, ground_truth))
            ADE_100s.append(ADE(y_100, ground_truth))
            FDE_10s.append(FDE(y_10, ground_truth))
            FDE_100s.append(FDE(y_100, ground_truth))
            # i skip the last batch since it could be of different size, could also just use something else than stack...
            if i >= num_batches-1:
                break
    measure_lists = ADE_ms, ADE_10s, ADE_100s, minADE_ks, FDE_ms, FDE_10s, FDE_100s, minFDE_ks
    measures = [torch.cat(measure, 0).mean().item()
                for measure in measure_lists]
    ADE_m, ADE_10, ADE_100, minADE_k, FDE_m, FDE_10, FDE_100, minFDE_k = measures
    names = "ADE_m", "ADE_10", "ADE_100", "minADE_k", "FDE_m", "FDE_10", "FDE_100", "minFDE_k"
    vals = [[val] for val in measures]
    df = pd.DataFrame(dict(zip(names, vals)))
    df.to_csv(output_path)


if __name__ == "__main__":
    # it would be beneficial to create a management class comparable to a jobcenter
    if False:  # example of how to evaluate a model on one data set
        ckpt_path = os.path.join(
            MODELS_PATH, "dim", "mymodel_d32", "ckpts", "model-100.pt")
        data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
        output_path = os.path.join(
            MODELS_PATH, "dim", "mymodel_d32", "eval_downloaded_100.csv")
        evaluate(ckpt_path, data_path, output_path,
                 mobilenet_num_classes=32, num_batches=20)

    if False:  # example of how to evaluate a DIM model on a dataset with several distributions
        ckpt_path = os.path.join(
            MODELS_PATH, "dim", "mymodel_d32", "ckpts", "model-200.pt")
        data_path_root = os.path.join(
            DATA_PATH, "mydistributions", "processed5", "val")
        output_path_raw = os.path.join(
            MODELS_PATH, "dim", "mymodel_d32", "eval_dists7.2_{}_200.csv")
        for dist in os.listdir(data_path_root):  # go over all distributions
            data_path = os.path.join(data_path_root, dist)
            output_path = output_path_raw.format(dist)
            evaluate(ckpt_path, data_path, output_path,
                     mobilenet_num_classes=32)
        evaluate(ckpt_path, data_path_root, output_path_raw.format(
            "all"), mobilenet_num_classes=32, num_batches=20)
