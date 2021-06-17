from .Patch_Attack.project_iter_attack import graph, clip_by_tensor
from ..distances import linf
from ..models import Model
from ..criteria import Criterion
from .base import MinimizationAttack

from typing import Union, Any, Optional, Callable, List

from torchvision import transforms as T
import torch
import numpy as np

class PatchAttack(MinimizationAttack):
    distance = linf
    def __init__(self, **kwargs):

        self.distance = linf
    def run(self, model:Model, inputs, criterion, max_epsilon:float=16.0, **kwargs: Any):
        image_width = 299
        # load image with 'RGB', wrap into a dataloader
        image = torch.from_numpy(inputs).permute(0, 3, 1, 2).float()


        # load ground_truth label
        ground_truth = torch.tensor(criterion.labels.raw)
        # generate adversarial image
        if torch.cuda.is_available():
            image = image.cuda()
            ground_truth = ground_truth.cuda()
        image_min = clip_by_tensor(image - max_epsilon / 255.0, 0.0, 1.0)
        image_max = clip_by_tensor(image + max_epsilon / 255.0, 0.0, 1.0)
        adv_image = graph(image, ground_truth, image_min, image_max, max_epsilon)
        adv_image = adv_image.cpu().permute(0, 2, 3, 1).numpy()
        return adv_image
