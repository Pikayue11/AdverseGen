from .L0_attack_back import CSattack
from typing import Union, Any, Optional, Callable, List
from typing_extensions import Literal

from .base import MinimizationAttack, get_is_adversarial, T, get_criterion
from .. import Distance
from ..models import Model
from ..criteria import Criterion
from ..distances import l1, l2, linf
from ..log_management import LogManagement


class CornerSearch(MinimizationAttack):
    distance = l1

    def __init__(
            self,
            **kwargs):
        self.args = {'type_attack': 'L0',
                     'n_iter': 1000,
                     'n_max': 100,
                     'kappa': -1,
                     'epsilon': -1,
                     'sparsity': 10,
                     'size_incr': 1}
        self.distance = linf
        self.args.update(kwargs)

    def run(self,
            model: Model,
            inputs: T,
            criterion: Union[Criterion, T],
            logger: LogManagement = None,
            **kwargs: Any
            ) -> T:
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        self.args['logger'] = logger
        attack = CSattack(model, self.args)
        new_trans_images, _ = attack.perturb(inputs, criterion.labels)
        return new_trans_images
