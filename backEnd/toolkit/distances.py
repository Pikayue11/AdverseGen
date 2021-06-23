from abc import ABC, abstractmethod
from typing import TypeVar
import eagerpy as ep
from skimage.metrics import _structural_similarity as ss
import numpy as np

from .devutils import flatten, atleast_kd

T = TypeVar("T")


class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T:
        ...

    @abstractmethod
    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        ...


class LpDistance(Distance):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, references: T, perturbed: T) -> T:
        """Calculates the distances from references to perturbed using the Lp norm.
        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
        Returns:
            A 1D tensor with the distances from references to perturbed.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        """Clips the perturbations to epsilon and returns the new perturbed
        Args:
            references: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
        Returns:
            A tenosr like perturbed but with the perturbation clipped to epsilon.
        """
        (x, y), restore_type = ep.astensors_(references, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
            return restore_type(x + clipped_perturbation)
        norms = ep.norms.lp(flatten(p), self.p, axis=-1)
        norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
        factor = epsilon / norms
        factor = ep.minimum(1, factor)  # clipping -> decreasing but not increasing
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("reducing L0 norms not yet supported")
        factor = atleast_kd(factor, x.ndim)
        clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)


class SSIMDistance(Distance):
    def __init__(self):
        ...

    def __repr__(self) -> str:
        return f"SSIMDistance"

    def __str__(self) -> str:
        return f"SSIM distance"

    def __call__(self, references: T, perturbed: T, *, win_size=None, gradient=False, data_range=None,
                 multichannel=False, gaussian_weights=False,
                 full=False, **kwargs) -> T:
        # only supports [n, m, c] image
        if not (isinstance(references, np.ndarray) and isinstance(perturbed, np.ndarray)):
            raise TypeError('only supports numpy array.')
        x = np.copy(references[0])
        y = np.copy(perturbed[0])
        if x.dtype != np.uint8:
            # change to 0-255
            if np.max(x) <= 1:
                x *= 255
            x.astype(np.uint8)

        if y.dtype != np.uint8:
            # change to 0-255
            if np.max(y) <= 1:
                y *= 255
            y.astype(np.uint8)

        if references.shape[2] > 1:
            return ss.structural_similarity(x, y, multichannel=True)
        else:
            return ss.structural_similarity(x, y, multichannel=False)

    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        raise NotImplementedError('reducing SSIM-norms not yet supported')


l0 = LpDistance(0)
l1 = LpDistance(1)
l2 = LpDistance(2)
linf = LpDistance(ep.inf)
ssim = SSIMDistance()


def get_distance(norm: str) -> Distance:
    if norm == 'l0':
        return l0
    elif norm == 'l1':
        return l1
    elif norm == 'l2':
        return l2
    elif norm == 'linf':
        return linf
    elif norm == 'ssim':
        return ssim
    else:
        raise NotImplementedError('no such norm')
