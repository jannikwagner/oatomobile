# OATomobile: A research framework for autonomous driving

  **[Overview](#overview)**
| **[Installation](#installation)**
| **[Baselines]**
| **[Paper]**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/oatomobile)
![PyPI version](https://badge.fury.io/py/oatomobile.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2006.14911-b31b1b.svg)](https://arxiv.org/abs/2006.14911)
[![GitHub license](https://img.shields.io/pypi/l/oatomobile)](./LICENSE)

OATomobile is a library for autonomous driving research.
OATomobile strives to expose simple, efficient, well-tuned and readable agents, that serve both as reference implementations of popular algorithms and as strong baselines, while still providing enough flexibility to do novel research.

## Overview

If you just want to get started using OATomobile quickly, the first thing to know about the framework is that we wrap [CARLA] towns and scenarios in OpenAI [gym]s:

```python
import oatomobile

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

while not done:
  # Selects a random action.
  action = environment.action_space.sample()
  observation, reward, done, info = environment.step(action)

  # Renders interactive display.
  environment.render(mode="human")

# Book-keeping: closes
environment.close()
```

[Baselines] can also be used out-of-the-box:

```python
# Rule-based agents.
import oatomobile.baselines.rulebased

agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)
action = agent.act(observation)

# Imitation-learners.
import torch
import oatomobile.baselines.torch

models = [oatomobile.baselines.torch.ImitativeModel() for _ in range(4)]
ckpts = ... # Paths to the model checkpoints.
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
agent = oatomobile.baselines.torch.RIPAgent(
  environment=environment,
  models=models,
  algorithm="WCM",
)
action = agent.act(observation)
```

## Installation

We have tested OATomobile on Python 3.5.

1. install python

    ```bash
    # install python
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod 777 Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc

    conda create -n py35 python=3.5
    conda activate py35
    ```

2. To install the core libraries (including [CARLA], the backend simulator):

    ```bash
    # The path to download CARLA 0.9.6.
    export CARLA_ROOT=...
    mkdir -p $CARLA_ROOT

    # Downloads hosted binaries.
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz

    # CARLA 0.9.6 installation.
    tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT

    # Installs CARLA 0.9.6 Python API.
    easy_install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg

    # install dependencies
    pip install -r $PWD/oatomobile/requirements.txt
    pip install -r $PWD/oatomobile/requirements2.txt
    ```

3. To install the OATomobile core API:

    ```bash
    pip install --upgrade pip setuptools
    pip install oatomobile
    ```

4. To install dependencies for our [PyTorch]- or [TensorFlow]-based agents:

    ```bash
    pip install oatomobile[torch]
    # and/or
    pip install oatomobile[tf]
    ```

## Citing OATomobile

If you use OATomobile in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@inproceedings{filos2020can,
    title={Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?},
    author={Filos, Angelos and
            Tigas, Panagiotis and
            McAllister, Rowan and
            Rhinehart, Nicholas and
            Levine, Sergey and
            Gal, Yarin},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```

[Baselines]: oatomobile/baselines/
[Examples]: examples/
[CARLA]: https://carla.readthedocs.io/
[Paper]: https://arxiv.org/abs/2006.14911
[TensorFlow]: https://tensorflow.org
[PyTorch]: http://pytorch.org
[gym]: https://github.com/openai/gym

## Modifications to OATomobile

I used OATomobile to generate data for my bachelor thesis and also modified it for that purpose. The scripts I wrote are very basic and not well designed. See ```generate_data.py``` for data generation. See ```train.py``` for training of a DIM model. A model can be trained as follows:

```bash
python train.py --dataset_dir=data/mydata/processed --output_dir=models/dim/mymodel_d32/ --num_epochs=201 --cuda_train_idx=0 --cuda_val_idx=0 --mobilenet_num_classes=32 --batch_size=512
```

. See ```evaluation.py``` for the evaluation of a trained DIM model. I also modified files of OATomobile such as ```oatomobile.datasets.carla.py```.
