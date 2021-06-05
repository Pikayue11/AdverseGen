from . import toolkit as tk
from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict
from typing_extensions import final, overload
from .toolkit import Model
import torch
T = TypeVar("T")

algo_dict = {"HopSkipJump": tk.attacks.HopSkipJump(), "CornerSearch": tk.attacks.CornerSearch()}

def attack_image(model:Model, inputs:T, algo:str, criterion: Any) -> Tuple[T, T]:
    model = tk.PyTorchModel(model, bounds=(0, 1))
    attaker = ImageAttacker(model, inputs, algo, criterion)
    return attaker.run()

class ImageAttacker:
    def __init__(
        self,
        model: Model,
        inputs: T,
        algo: str,
        criterion: Any):
        # get algorithm
        self.algorithm = algo_dict[algo]
        self.inputs = inputs
        self.inputs_torch = torch.from_numpy(inputs).permute(0, 3, 1, 2).float()
        self.victim_model = model
        self.criterion = criterion


    def run(self) -> Tuple[T, T]:
        epsilons = [1.0]
        _, advs, success = self.algorithm(self.victim_model, self.inputs, self.criterion, epsilons=epsilons)
        return advs, success







