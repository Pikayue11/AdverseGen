from .L0_attack_back import CSattack
from typing import Union, Any, Optional, Callable, List
from typing_extensions import Literal
import torch
from torchvision import transforms
import numpy as np
from .base import FixedEpsilonAttack, get_is_adversarial, T, get_criterion, MinimizationAttack
from .. import Distance
from ..models import Model
from ..criteria import Criterion, TargetedMisclassification
from ..distances import l1, l2, linf

from .SSIM_attack_back import PQP

from ..log_management import LogManagement

class PQPAttack(MinimizationAttack):
    distance = l1

    def __init__(
            self,
            **args):
        self.args = {'loss_goal': None,
                     'minimize_loss': False,
                     'ssim_th': 0.95,
                     'M': 66.,
                     'N': 20,
                     'delta': 1,
                     'k_max': 20,
                     'print_every': 100}
        self.distance = linf

    def run(self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, T],
            logger: LogManagement=None,
            **kwargs: Any
            ) -> T:
        def query_fun(img):  # img is [32,32,3], the elements are from [0 to 255]
            with torch.no_grad():
                # change img from [32,32,3] to [1,32,32,3]
                img = np.expand_dims(img / 255, axis=0)
                output = torch.nn.Softmax(dim=1)(torch.from_numpy(model(img))).squeeze().numpy()
            return output

        criterion = get_criterion(criterion)
        if not isinstance(criterion, TargetedMisclassification):
            output = model(inputs=inputs)
            target_class = np.squeeze(np.argsort(output))[-2]
            criterion = TargetedMisclassification(labels=criterion.labels, target_classes=np.array([target_class]))

        is_adversarial = get_is_adversarial(criterion, model)
        mean = lambda x: np.asarray(x).mean()
        success, ssim, psnr, NQ = [], [], [], []
        adv_images = np.copy(inputs)
        labels = criterion.labels
        target_classes = criterion.target_classes
        num_imgs = inputs.shape[0]
        for i in range(num_imgs):
            img = (inputs[i] * 255).astype(np.uint8)
            label = labels[i]

            # init log
            ori_img = np.expand_dims(np.copy(img), axis=0)
            myLog = logger

            # start attack
            newImg, success_, ssim_, psnr_, NQ_, _ = PQP(labels[i].raw, query_fun=query_fun, or_img=img,
                                                         target=target_classes[i].raw, logger=myLog)
            newImg = np.uint8(newImg)
            adv_images[i] = newImg / 255
            if (success_):
                success.append(1)
            else:
                success.append(0)
            ssim.append(ssim_)
            psnr.append(psnr_)
            NQ.append(NQ_)

        return adv_images