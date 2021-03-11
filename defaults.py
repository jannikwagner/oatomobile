import os

os.environ["CARLA_ROOT"]=os.path.join(os.curdir,"carla")

PATH = os.path.join(os.getcwd())
DATA_PATH = os.path.join(PATH, "data")
MODELS_PATH = os.path.join(PATH, "models")
