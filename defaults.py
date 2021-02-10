import os
import torch

device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
os.environ["CARLA_ROOT"]="/home/jannik_wagner/carla"

import carla

WEATHERS = (
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.Default,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.SoftRainSunset,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetSunset
)
PATH = os.path.join(os.getcwd())
DATA_PATH = os.path.join(PATH, "data")
MODELS_PATH = os.path.join(PATH, "models")
