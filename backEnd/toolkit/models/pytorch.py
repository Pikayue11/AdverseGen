from typing import Any, cast, Union
import warnings
import eagerpy as ep

from ..types import BoundsInput, Preprocessing

from .base import ModelWithPreprocessing, T


def get_device(device: Any) -> Any:
    import torch

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


class PyTorchModel(ModelWithPreprocessing):

    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        device: Any = None,
        preprocessing: Preprocessing = None,
    ):
        import torch

        if not isinstance(model, torch.nn.Module):
            raise ValueError("expected model to be a torch.nn.Module instance")

        if model.training:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The PyTorch model is in training mode and therefore might"
                    " not be deterministic. Call the eval() method to set it in"
                    " evaluation mode if this is not intended."
                )

        device = get_device(device)
        model = model.to(device)
        dummy = ep.torch.zeros(0, device=device)
        self.data_format = "channels_first"
        self.device = device
        self.convert = {"Pytorch": self.torch2torch, "TensorFlow": self.tensor2torch, "Numpy": self.numpy2torch}
        self.type = 'Numpy'

        # we need to make sure the output only requires_grad if the input does
        def _model(x: T) -> T:
            x = self.convert[self.type](x)
            x = x.permute(0, 3, 1, 2).float()
            with torch.set_grad_enabled(x.requires_grad):
                result = cast(torch.Tensor, model(x.to(device)))
            result = result.cpu()
            result = self.convert[self.type](result, revert=True)
            return result

        super().__init__(
            _model, bounds=bounds, dummy=dummy, preprocessing=preprocessing
        )

    def type_convert(self, inputs: T, revert=False) -> T:
        ...

    def torch2torch(self, input, revert=False) -> T:
        return input

    def numpy2torch(self, input, revert=False) -> T:
        import torch
        if revert:
            return input.numpy()
        else:
            return torch.from_numpy(input)

    def tensor2torch(self, input, revert=False) -> T:
        import torch
        import tensorflow as tf
        input_np = input.numpy()
        if revert:
            return tf.convert_to_tensor(input_np)
        else:
            return torch.from_numpy(input_np)