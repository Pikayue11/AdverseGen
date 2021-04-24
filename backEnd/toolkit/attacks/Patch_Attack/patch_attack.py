from typing import TypeVar, Any, Union
from typing_extensions import Literal

from PIL import Image
import torchvision.models as models

from project_iter_attack import clip_by_tensor
from project_iter_attack import graph
from Normalize import *
import torch
import numpy as np
from torchvision import transforms as T
from toolkit.models import Model

from ..base import FixedEpsilonAttack, get_is_adversarial,get_criterion
from toolkit.criteria import Criterion, Misclassification, TargetedMisclassification
import eagerpy as ep

'''
patch_attack (untargeted attack version)
:param image: one input image, should be in RGB format
:param ground_truth: correct label of image
:param victim_model: res152, inc_v3, resnext50_32x4d, dense161
'''

E = TypeVar("E")

class PatchAttack(FixedEpsilonAttack):
    def __init__(
            self,

            constraint: Union[Literal["linf"], Literal["l2"]] = "linf"
                 ):
        self.constraint = constraint

    def run(
        self,
        model: Model,
        inputs: E,
        criterion: Union[Criterion, E],
        *,
        epsilon: float,
        **kwargs: Any) -> E:

        originals, restore_type = ep.astensor_(inputs)
        del inputs
        criterion = get_criterion(criterion)
        targeted = False
        if isinstance(criterion, Misclassification):
            targeted = False
            ground_truth = criterion.labels
        elif isinstance(criterion, TargetedMisclassification):
            targeted = True
            ground_truth = criterion.target_classes
        else:
            raise ValueError("unsupported criterion")
        is_adversarial = get_is_adversarial(criterion, model)

        inputs_min = clip_by_tensor(originals - epsilon / 255.0, 0.0, 1.0)
        inputs_max = clip_by_tensor(originals + epsilon / 255.0, 0.0, 1.0)
        adv_image = graph(originals, ground_truth, inputs_min, inputs_max, epsilon)
        return restore_type(adv_image)





def patch_attack(image, ground_truth, victim_model: Model):
    # hyperparameter list #
    image_width = 299
    max_epsilon = 16.0
    # load image with 'RGB', wrap into a dataloader
    transforms = T.Compose([T.CenterCrop(image_width), T.ToTensor()])
    image = transforms(image.convert('RGB'))
    image = image.unsqueeze(0)
    # load ground_truth label
    ground_truth = torch.tensor([ground_truth])
    # generate adversarial image
    if torch.cuda.is_available():
        image = image.cuda()
        ground_truth = ground_truth.cuda()
    image_min = clip_by_tensor(image - max_epsilon / 255.0, 0.0, 1.0)
    image_max = clip_by_tensor(image + max_epsilon / 255.0, 0.0, 1.0)
    adv_image = graph(image, ground_truth, image_min, image_max, max_epsilon)
    # check whether attack succeed
    with torch.no_grad():
        success_num = (victim_model(adv_image).argmax(1) != ground_truth).detach().sum().cpu()
    # print result
    print("Successful Attack Num: %d" % success_num)


if __name__ == '__main__':
    image = Image.open('../dataset/images/0f9d8c86a9f38020.png').convert('RGB')
    ground_truth = 760 - 1
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet dataset preprocess
    std = np.array([0.229, 0.224, 0.225])  # ImageNet dataset preprocess
    if torch.cuda.is_available():
        res152 = torch.nn.Sequential(Normalize(mean, std), models.resnet152(pretrained=True).eval().cuda())
    else:
        res152 = torch.nn.Sequential(Normalize(mean, std), models.resnet152(pretrained=True).eval())
    patch_attack(image, ground_truth, res152)
