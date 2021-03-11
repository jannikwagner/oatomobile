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


def getDIM(path=None,device="cpu", mobilenet_num_classes=128) -> ImitativeModel:
    if path is None:
        path = os.path.join(MODELS_PATH, "dim", "9", "ckpts", "model-96.pt")
    model = ImitativeModel(mobilenet_num_classes=mobilenet_num_classes)
    x = torch.load(path)
    model.load_state_dict(x)
    model.eval().to(device)
    return model


def get_agent_fn(model):
    def agent_fn(environment):
        return DIMAgent(environment, model=model)
    return agent_fn

def transform(model, batch,
            device) -> Mapping[str, torch.Tensor]:
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
    return (pred - ground_truth).square().sum(axis=-1).sqrt().mean(axis=-1)

def FDE(pred: torch.Tensor, ground_truth: torch.Tensor):
    (pred[:,-1,:] - ground_truth[:,-1,:]).square().sum(axis=-1).sqrt()

def minADE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_ADE = [ADE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_ADE).min(axis=0)

def minFDE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_FDE = [FDE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_FDE).min(axis=0)

def evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=128):
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
    ADE_10s = []
    ADE_100s = []
    minADE_ks = []
    FDE_ms = []
    FDE_10s = []
    FDE_100s = []
    minFDE_ks = []
    with tqdm.tqdm(dataloader) as pbar:
        for batch in pbar:  # gives errors with num_workers > 1
            # Prepares the batch.

            batch = transform(model, batch, device)
            ground_truth = batch["player_future"]
            samples = [model.sample(**batch) for _ in range(k)]
            mean = model.mean(**batch)
            y_10 = model.forward(10,**batch)
            y_100 = model.forward(100,**batch)
            ADE_ms.append(ADE(mean, ground_truth))
            ADE_10s.append(ADE(y_10, ground_truth))
            ADE_100s.append(ADE(y_100, ground_truth))
            minADE_ks.append(minADE(samples, ground_truth))
            FDE_ms.append(FDE(mean, ground_truth))
            FDE_10s.append(FDE(y_10, ground_truth))
            FDE_100s.append(FDE(y_100, ground_truth))
            minFDE_ks.append(minFDE(samples, ground_truth))
    measure_lists = ADE_ms,ADE_10s,ADE_100s,minADE_ks,FDE_ms,FDE_10s,FDE_100s,minFDE_ks
    measures = [torch.stack(measure).mean().item() for measure in measure_lists]
    ADE_m,ADE_10,ADE_100,minADE_k,FDE_m,FDE_10,FDE_100,minFDE_k = measures
    names = "ADE_m","ADE_10","ADE_100","minADE_k","FDE_m","FDE_10","FDE_100","minFDE_k"
    vals = [[val] for val in measures]
    df = pd.DataFrame(dict(zip(names, vals)))
    df.to_csv(output_path)

if __name__ == "__main__":
    ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d128", "ckpts","model-100.pt")
    data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d128", "eval_100.csv")
    evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=128)
    