import numpy as np
import os
import 


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

