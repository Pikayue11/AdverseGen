"""Implementation of sample attack."""
import os
import torch
import torchvision.models as models
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
import numpy as np
from .Normalize import Normalize
import argparse

from ...log_management import LogManagement

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='dataset/dev_dataset.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input directory with images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=330, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--amplification", type=float, default=10.0, help="To amplifythe step size.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")

opt = parser.parse_args()

# hyperparamter list
image_width = 299
image_resize = 330
prob = 0.7


def get_device(device):
    import torch

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


device = get_device(None)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False

transforms = T.Compose([T.CenterCrop(image_width), T.ToTensor()])


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)

    stack_kern = torch.tensor(stack_kern).to(device)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding=(kern_size, kern_size), groups=3)
    return x


stack_kern, kern_size = project_kern(3)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(x, gt, x_min, x_max, max_epsilon, logger: LogManagement = None):
    eps = max_epsilon / 255.0
    # Hyperparameter list
    num_iter = 10
    amplification_factor = 10.0
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet dataset preprocess
    std = np.array([0.229, 0.224, 0.225])  # ImageNet dataset preprocess
    # start attack
    alpha = eps / num_iter
    alpha_beta = alpha * amplification_factor
    gamma = alpha_beta
    amplification = 0.0
    device = get_device(None)

    model = torch.nn.Sequential(Normalize(mean, std), models.inception_v3(pretrained=True).eval().to(device))
    x = x.to(device)

    x.requires_grad = True

    for i in range(num_iter):
        zero_gradients(x)
        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)
        loss.backward()
        noise = x.grad.data

        # MI-FGSM
        # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
        # noise = momentum * grad + noise
        # grad = noise

        amplification += alpha_beta * torch.sign(noise)
        cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
        projection = gamma * torch.sign(project_noise(cut_noise, stack_kern, kern_size))
        amplification += projection

        # x = x + alpha * torch.sign(noise)
        x = x + alpha_beta * torch.sign(noise) + projection
        x = clip_by_tensor(x, x_min, x_max)
        x = V(x, requires_grad=True)
        if logger is not None:
            logger.imgUpdate(x.detach().cpu().permute(0, 2, 3, 1).numpy(), i)
    logger.logEnd(x.detach().cpu().permute(0, 2, 3, 1).numpy())
    return x.detach()


def input_diversity(input_tensor):
    rnd = torch.randint(opt.image_width, opt.image_resize, ())
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = opt.image_resize - rnd
    w_rem = opt.image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [opt.image_resize, opt.image_resize])
    return padded if torch.rand(()) < opt.prob else input_tensor
