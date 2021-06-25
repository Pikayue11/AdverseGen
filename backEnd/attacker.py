from . import toolkit as tk
from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict
from typing_extensions import final, overload
from .toolkit import Model
import torch
from .toolkit.model_importer import modelImporter
import numpy as np
from .toolkit.criteria import TargetedMisclassification
from .toolkit.log_management import LogManagement
from .toolkit.distances import Distance, get_distance

T = TypeVar("T")

algo_dict = {"HopSkipJump": tk.attacks.HopSkipJump, # l2 & l8, when select decision based

             "CornerSearch": tk.attacks.CornerSearch,   # l0, select <=3, or l8 or l2

             "PQPAttack": tk.attacks.PQPAttack,     # ssim, has ssim

             "PatchAttack": tk.attacks.PatchAttack, # l8

             "LeBA": tk.attacks.LeBA}   # l2

model_avail = {"CIFAR-10": ['Resnet18'], "ImageNet": ['ResNet152']}

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

    # map_cons = {'l0': 1, 'l8': 0, 'ssim': 0, 'l2': 0}  # ch:l0, la:l8, st:ssim, eu:l2
    # map_value = {'l0': '3', 'l8': '', 'ssim': '', 'l2': ''}
    # 'Score based', 'Decision based'
    def set_algo(self, map_constraints, based):
        if based == 'Decision based':
            self.algo = algo_dict['HopSkipJump']()
            return
        if map_constraints['l0'] or (map_constraints['l0'] and map_constraints['l8']):
            self.algo = algo_dict['CornerSearch']()
        elif map_constraints['l8']:
            self.algo = algo_dict['PatchAttack']()
        elif map_constraints['ssim'] == 'SSIM':
            self.algo = algo_dict['PQPAttack']()
        elif map_constraints['l2']:
            self.algo = algo_dict['LeBA']()
        else:
            raise ValueError('No Algorithm implemented.')

    def update_model(self, mname: str):
        if self.fmodel[0] is None or self.fmodel[1] != mname:
            self.fmodel[0], self.fmodel[1] = modelImporter(mname)

    def run(self, input: T, label: T, target_label: T,  map_constraints, map_value, based, verbose: bool=True) -> Tuple[T, T, int, T]:
        self.set_algo(map_constraints, based)
        distance = get_distance(evaluation)
        if verbose:
            logger = LogManagement(input, label[0], self.fmodel, distance, target=target_label, databaseName=self.database)
        else:
            logger = None

        if target_label is None:
            _, advs, success = self.algo(self.fmodel[0], input, label, epsilons=None, logger=logger)
        else:
            if not isinstance(label, list):
                label = [label]
            if not isinstance(target_label, list):
                target_label = [target_label]
            criterion = TargetedMisclassification(labels=label, target_classes=target_label)
            _, advs, success = self.algo(self.fmodel[0], input, criterion, epsilons=None, logger=logger)
        adv_id = self.get_label(self.fmodel[1], advs)
        return advs, adv_id, 0, success