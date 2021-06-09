from . import toolkit as tk
from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict
from typing_extensions import final, overload
from .toolkit import Model
import torch
from .toolkit.model_importer import modelImporter
import numpy as np
from .toolkit.criteria import TargetedMisclassification

T = TypeVar("T")

algo_dict = {"HopSkipJump": tk.attacks.HopSkipJump, "CornerSearch": tk.attacks.CornerSearch,
             "PQPAttack": tk.attacks.PQPAttack}

model_avail = {"CIFAR-10": ['Resnet18'], "ImageNet": []}

class ImageAttacker:
    def __init__(
            self,
            database: str=None):
        # set database
        self.database = database

        self.fmodel: List[Optional[Model, None], str] = [None, '']
        self.algo = None

    def get_available_model(self) -> list:
        return model_avail[self.database]

    def set_database(self, database: str):
        self.database = database

    def get_label(self, mname: str, input: T):
        self.update_model(mname)
        self.fmodel[0].set_type('Numpy')
        with torch.no_grad():
            y_test = self.fmodel[0](input)
        labels = np.zeros(input.shape[0], dtype=np.int64)
        for i in range(input.shape[0]):
            labels[i] = y_test[i].argmax()
        return labels

    def set_algo(self, constraint: str):
        if constraint == 'L0':
            self.algo = algo_dict['CornerSearch']()
        elif constraint == 'SSIM':
            self.algo = algo_dict['PQPAttack']()
        elif constraint == 'Decision-based':
            self.algo = algo_dict['HopSkipJump']()
        else:
            raise ValueError('No Algorithm implemented.')

    def update_model(self, mname: str):
        if self.fmodel[0] is None or self.fmodel[1] != mname:
            self.fmodel[0], self.fmodel[1] = modelImporter(mname)

    def run(self, input: T, label: T, target_label: T, eval: str) -> Tuple[T, T, int, T]:
        self.set_algo(eval)
        if target_label is None:
            _, advs, success = self.algo(self.fmodel[0], input, label, epsilons=None)
        else:
            if not isinstance(label, list):
                label = [label]
            if not isinstance(target_label, list):
                target_label = [target_label]
            criterion = TargetedMisclassification(labels=label, target_classes=target_label)
            _, advs, success = self.algo(self.fmodel[0], input, criterion, epsilons=None)
        adv_id = self.get_label(self.fmodel[1], advs)
        return advs, adv_id, 0, success
