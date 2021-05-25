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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()


def getDIM(path=None,device="cpu", mobilenet_num_classes=128) -> ImitativeModel:
    model = ImitativeModel(mobilenet_num_classes=mobilenet_num_classes)
    if path is not None:
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
    return (pred[:,:,:2] - ground_truth[:,:,:2]).square().sum(axis=-1).sqrt().mean(axis=-1)

def FDE(pred: torch.Tensor, ground_truth: torch.Tensor):
    return (pred[:,-1,:2] - ground_truth[:,-1,:2]).square().sum(axis=-1).sqrt()

def minADE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_ADE = [ADE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_ADE).min(axis=0)[0]

def minFDE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_FDE = [FDE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_FDE).min(axis=0)[0]

def evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=128,num_batches=np.inf,k=5):
    model = getDIM(ckpt_path, device, mobilenet_num_classes)
    modalities = (
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "player_future",
        "velocity",
    )
    batch_size = 64
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
    ADE_ms = []
    ADE_10s = []
    ADE_100s = []
    minADE_ks = []
    FDE_ms = []
    FDE_10s = []
    FDE_100s = []
    minFDE_ks = []
    torch.cuda.empty_cache()
    with tqdm.tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):  # gives errors with num_workers > 1
            #if True:
                # Prepares the batch.

                batch = transform(model, batch, device)
                ground_truth = batch["player_future"]
                lengths=(ground_truth[:,3,:]-ground_truth[:,0,:]).square().sum(axis=1).sqrt()
                ls = lengths#[lengths>torch.zeros(lengths.size(),device="cuda")+0.1]
                print(ls.mean())
                y_samples = [model.sample(**batch).detach() for _ in range(k)]
                minADE_ks.append(minADE(y_samples, ground_truth))
                minFDE_ks.append(minFDE(y_samples, ground_truth))
                del y_samples
                y_mean = model.mean(**batch).detach()
                ADE_ms.append(ADE(y_mean, ground_truth))
                FDE_ms.append(FDE(y_mean, ground_truth))
                del y_mean
                # y_10 = model.forward(10,**batch).detach()
                # y_100 = model.forward(100,**batch).detach()
                # ADE_10s.append(ADE(y_10, ground_truth))
                # ADE_100s.append(ADE(y_100, ground_truth))
                # FDE_10s.append(FDE(y_10, ground_truth))
                # FDE_100s.append(FDE(y_100, ground_truth))
                if i >= num_batches-1:  # i skip the last batch since it could be of different size, could also just use something else than stack...
                    break
    measure_lists = ADE_ms,ADE_10s,ADE_100s,minADE_ks,FDE_ms,FDE_10s,FDE_100s,minFDE_ks
    measures = [torch.cat(measure, 0).mean().item() if len(measure) > 0 else np.nan for measure in measure_lists]
    ADE_m,ADE_10,ADE_100,minADE_k,FDE_m,FDE_10,FDE_100,minFDE_k = measures
    names = "ADE_m","ADE_10","ADE_100","minADE_{}".format(k),"FDE_m","FDE_10","FDE_100","minFDE_{}".format(k)
    vals = [val for val in measures]
    d = dict(zip(names, vals))
    df = pd.DataFrame({0:d}).transpose()
    # df.to_csv(output_path)
    return d

def evaluate_wrapper(ckpts_path, data_path, output_path, mobilenet_num_classes=128,num_batches=np.inf,k=5):
    dfs = {}
    for x in tqdm.tqdm(sorted(os.listdir(ckpts_path),key=lambda x: int(x.replace("model-","").replace(".pt","")))):
        if x.startswith("model-") and x.endswith(".pt"):
            epoch = int(x.replace("model-","").replace(".pt",""))
            ckpt_path = os.path.join(ckpts_path,x)
            df = evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes,num_batches,k)
            dfs[epoch] = df
    df = pd.DataFrame(dfs).transpose()

    plot(df, k, output_path)
    df.to_csv(output_path)

def plot(df, k, output_path):
    plt.clf()
    plt.plot(df.index,df["FDE_m"],label="FDE")
    plt.plot(df.index,df["ADE_m"],label="ADE")
    plt.plot(df.index,df["minFDE_{}".format(k)],label="minFDE_{}".format(k))
    plt.plot(df.index,df["minADE_{}".format(k)],label="minADE_{}".format(k))
    plt.xlabel("epoch")
    plt.ylabel("meter")
    plt.ylim(0)
    plt.legend()
    plt.savefig(output_path+".pdf",bbox_inches='tight')
    # print(dfs)

def plot_wrapper(csv_path, k):
    df = pd.read_csv(csv_path)
    plot(df,k,csv_path)
    

if __name__ == "__main__":
    
    # dists on dists
    ckpt_path = os.path.join(MODELS_PATH, "dim","dists10_d128", "ckpts")
    data_path_root = os.path.join(DATA_PATH, "dists10", "processed5", "val")
    output_path_raw = os.path.join(MODELS_PATH, "dim", "dists10_d128", "eval_dists10_{}.csv")
    # for dist in os.listdir(data_path_root):
    #     data_path = os.path.join(data_path_root, dist)
    #     output_path = output_path_raw.format(dist)
    #     evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=32)
    evaluate_wrapper(ckpt_path, data_path_root, output_path_raw.format("all"), mobilenet_num_classes=128,k=5)
    

    ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d32", "ckpts")
    data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d32", "eval_downloaded.csv")
    evaluate_wrapper(ckpt_path, data_path, output_path, mobilenet_num_classes=32,k=5)
    
    # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d64", "ckpts")
    # data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    # output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d64", "eval_downloaded.csv")
    # evaluate_wrapper(ckpt_path, data_path, output_path, mobilenet_num_classes=64,k=5)
    
    # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d128", "ckpts")
    # data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    # output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d128", "eval_downloaded.csv")
    # evaluate_wrapper(ckpt_path, data_path, output_path, mobilenet_num_classes=128,k=5)
    


    # ckpt_path = os.path.join(MODELS_PATH, "dim","dists7.2_moving_d32", "ckpts","model-40.pt")
    # data_path_root = os.path.join(DATA_PATH, "dists7.2", "processed5_moving", "val")
    # output_path_raw = os.path.join(MODELS_PATH, "dim", "dists7.2_moving_d32", "eval_dists7.2_moving_{}_40.csv")
    # for dist in os.listdir(data_path_root):
    #     data_path = os.path.join(data_path_root, dist)
    #     output_path = output_path_raw.format(dist)
    #     evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=32)
    # evaluate(ckpt_path, data_path_root, output_path_raw.format("all"), mobilenet_num_classes=32,num_batches=20)

    # # dowloaded on dists
    # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d32", "ckpts","model-200.pt")
    # data_path_root = os.path.join(DATA_PATH, "dists7.2", "processed5_moving3", "val")
    # output_path_raw = os.path.join(MODELS_PATH, "dim", "downloaded_d32", "eval_dists7.2_moving3_{}_200.csv")
    # for dist in os.listdir(data_path_root):
    #     data_path = os.path.join(data_path_root, dist)
    #     output_path = output_path_raw.format(dist)
    #     evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=32)
    # evaluate(ckpt_path, data_path_root, output_path_raw.format("all"), mobilenet_num_classes=32,num_batches=20)

    # # downloaded on downloaded
    # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d32", "ckpts","model-100.pt")
    # data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    # output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d32", "eval_downloaded_100.csv")
    # evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=32,num_batches=20)

    # # downloaded on downloaded
    # ckpt_path = os.path.join(MODELS_PATH, "dim","downloaded_d32", "ckpts","model-4.pt")
    # data_path = os.path.join(DATA_PATH, "downloaded", "processed", "val")
    # output_path = os.path.join(MODELS_PATH, "dim", "downloaded_d32", "eval_downloaded_4.csv")
    # evaluate(ckpt_path, data_path, output_path, mobilenet_num_classes=32,num_batches=20)
    