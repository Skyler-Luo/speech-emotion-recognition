import torch.nn as nn

from models.mn import get_model as get_mobilenet
from models.dymn import get_model as get_dymn
from utils.utils import NAME_TO_WIDTH


class EnsemblerModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        all_out = None
        for m in self.models:
            out, _ = m(x)
            all_out = out if all_out is None else all_out + out
        all_out = all_out / len(self.models)
        return all_out, all_out


def get_ensemble_model(model_names):
    models = []
    for name in model_names:
        if name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(name), pretrained_name=name)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(name), pretrained_name=name)
        models.append(model)
    return EnsemblerModel(models)
