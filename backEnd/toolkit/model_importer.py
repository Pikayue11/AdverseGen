from .models import *
import torch
import os
import torchvision.models as models

def ResNet18Init() -> Model:
    model = ResNet18()
    if torch.cuda.is_available():
        ckpt = torch.load(r'./backEnd/toolkit/models/model_weight/ResNet18.pt')
    else:
        ckpt = torch.load(r'./backEnd/toolkit/models/model_weight/ResNet18.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    return PyTorchModel(model, bounds=(0, 1))

def ResNet152Init() -> Model:
    model = models.resnet152(pretrained=True)
    model.eval()
    return PyTorchModel(model, bounds=(0, 1))

model_dict = {"ResNet18": ResNet18Init, "ResNet152": ResNet152Init}


def modelImporter(mname: str):
    if mname in model_dict.keys():
        init_func = model_dict[mname]
        return init_func(), mname
    else:
        raise ValueError('model name not found.')