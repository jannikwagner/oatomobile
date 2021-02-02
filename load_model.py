import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import imageio
import glob
import torch
from typing import Mapping
from defaults import MODELS_PATH, DATA_PATH, device

import oatomobile
import oatomobile.envs
from oatomobile.datasets.carla import CARLADataset
import oatomobile.baselines.torch.dim.train as train
from oatomobile.core.dataset import Episode
from oatomobile.baselines.rulebased.autopilot.agent import AutopilotAgent
from oatomobile.torch.networks.perception import MobileNetV2
from oatomobile.baselines.torch.dim.model import ImitativeModel
from oatomobile.baselines.torch.dim.agent import DIMAgent
import carla
import itertools


def getDIM(path=None, output_shape=(4,2), device=device):
    torch.cuda.empty_cache()
    model = ImitativeModel(output_shape=output_shape).to(device).eval()
    if path:
        x = torch.load(path)
        model.load_state_dict(x)
    return model


def get_agent_fn(model):
    def agent_fn(environment):
        return DIMAgent(environment, model=model)
    return agent_fn


def test_eval(path=os.path.join(MODELS_PATH, "dim", "dists2", "ckpts", "model-200.pt")):
    model = getDIM(path).eval()
    fun(agent_fn=get_agent_fn(model))


def fun(sensors=(
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
        agent_fn=AutopilotAgent):
    
    CARLADataset.collect("Town01", os.path.join(DATA_PATH, "dim"), 100, 100, 1000, None, None, sensors, False, agent_fn)

def ADE(y1, y2):
    y1 = y1[..., :2]
    y2 = y2[..., :2]
    ade = torch.mean(torch.sqrt(torch.sum(torch.square(y1-y2),dim=-1)))
    return ade

def FDE(y1, y2):
    y1 = y1[..., -1, :2]
    y2 = y2[..., -1, :2]
    fde = torch.mean(torch.sqrt(torch.sum(torch.square(y1-y2),dim=-1)))
    return fde

def minFDEk(model, trans_batch, k):
    tfp = trans_batch["player_future"]
    samples = [model.sample(**trans_batch) for _ in range(k)]
    minFDEs = []
    for i in range(tfp.size()[0]):
        FDEs = [FDE(sample[i], tfp[i]) for sample in samples]
        minFDEs.append(min(FDEs))
    return torch.Tensor(minFDEs).mean()

def minADEk(model, trans_batch, k):
    tfp = trans_batch["player_future"]
    samples = [model.sample(**trans_batch) for _ in range(k)]
    minADEs = []
    for i in range(tfp.size()[0]):
        ADEs = [ADE(sample[i], tfp[i]) for sample in samples]
        minADEs.append(min(ADEs))
    return torch.Tensor(minADEs).mean()

def test_model():
    batch_size=32
    path=os.path.join(MODELS_PATH, "dim", "dists2", "ckpts", "model-200.pt")
    model = getDIM(path)
    modalities = (
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "player_future",
        "velocity",
    )

    dataset_dir = os.path.join(DATA_PATH, "dists", "processed", "test")
    measures = {}
    for dist in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, dist)):
            measures[dist] = torch.zeros(10)
            dataset_test = CARLADataset.as_torch(
                dataset_dir=os.path.join(dataset_dir, dist),
                modalities=modalities,
            )
            dataloader_test = torch.utils.data.DataLoader(  # errors with num_workers > 1
                dataset_test,
                batch_size=batch_size,
                shuffle=True,
            )
            for i, batch in enumerate(tqdm.tqdm(dataloader_test)):
                batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
                # Preprocesses batch for the model.
                batch = model.transform(batch)
                tfp = batch["player_future"]
                y_ml100 = model(100, **batch)
                y_ml10 = model(10, **batch)
                y_m = model.mean(**batch)
                measures[dist] += torch.Tensor([ADE(y_ml100,tfp),ADE(y_ml10,tfp),ADE(y_m,tfp),minADEk(model,batch,1),minADEk(model,batch,5),FDE(y_ml100,tfp),FDE(y_ml10,tfp),FDE(y_m,tfp),minFDEk(model,batch,1),minFDEk(model,batch,5)]).to("cpu")
                
            measures[dist] /= i
            print(measures[dist])

    with open(os.path.join(dataset_dir, "measures.csv"),"w") as f:
        f.write("data,ml100ADE,y_ml10ADE,y_mADE,minADE1,minADE5,y_ml100FDE,y_ml10FDE,y_mFDE,minFDE1,minFDE5\n")
        for dist in measures:
            x = dist+","+",".join(str(x.item()) for x in measures[dist]) + "\n"
            print(x)
            f.write(x)


if __name__ == "__main__":
    test_eval()
