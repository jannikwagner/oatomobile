from defaults import PATH, MODELS_PATH, DATA_PATH

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


def getDIM(path=None, mobilenet_num_classes=128):
    if path is None:
        path = os.path.join(MODELS_PATH, "dim", "9", "ckpts", "model-96.pt")
    model = ImitativeModel(mobilenet_num_classes=mobilenet_num_classes)
    x = torch.load(path)
    model.load_state_dict(x)
    return model


def get_agent_fn(model):
    def agent_fn(environment):
        return DIMAgent(environment, model=model)
    return agent_fn

def evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=128):
    model = getDIM(ckpt_path, mobilenet_num_classes)
    
    
