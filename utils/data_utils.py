import os
from math import log

import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms

from args import args


class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size


class MSDSDataset(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        with h5py.File(f'{root}_labels.hdf5', 'r') as f:
            self.labels = torch.from_numpy(np.asarray(f['labels']))

    def __getitem__(self, index):
        # Open path as file to avoid ResourceWarning
        with open(f'{self.root}/0/{index:05}.png', 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        return self.labels.shape[0]


def load_dataset():
    """Load dataset.

    Returns:
        a torch dataset and its associated information.
    """

    if args.data == 'celeba32':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'celeba')
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.CelebA(root,
                                    split='train',
                                    transform=transform,
                                    download=True)
        test_set = datasets.CelebA(root,
                                   split='test',
                                   transform=transform,
                                   download=True)

    elif args.data == 'mnist32':
        data_info = DataInfo(args.data, 1, 32)
        root = os.path.join(args.data_path, 'mnist')
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(root,
                                   train=True,
                                   transform=transform,
                                   download=True)
        test_set = datasets.MNIST(root,
                                  train=False,
                                  transform=transform,
                                  download=True)


    elif args.data == 'cifar10':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'cifar10')
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(root,
                                     train=True,
                                     transform=transform,
                                     download=True)
        test_set = datasets.CIFAR10(root,
                                    train=False,
                                    transform=transform,
                                    download=True)

    elif args.data == 'chair600':
        data_info = DataInfo(args.data, 3, 32)
        root = os.path.join(args.data_path, 'chair600')
        transform = transforms.Compose([
            transforms.CenterCrop(300),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(root, transform=transform)
        train_set, test_set = data.random_split(
            train_set, [77730, 8636],
            generator=torch.Generator().manual_seed(0))

    elif args.data == 'msds1':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.ToTensor()
        train_set = MSDSDataset(os.path.join(args.data_path, 'msds1/train'),
                                transform=transform)
        test_set = MSDSDataset(os.path.join(args.data_path, 'msds1/test'),
                               transform=transform)

    elif args.data == 'msds2':
        data_info = DataInfo(args.data, 3, 32)
        transform = transforms.ToTensor()
        train_set = MSDSDataset(os.path.join(args.data_path, 'msds2/train'),
                                transform=transform)
        test_set = MSDSDataset(os.path.join(args.data_path, 'msds2/test'),
                               transform=transform)

    else:
        raise ValueError(f'Unknown data: {args.data}')

    assert data_info.channel == args.nchannels
    assert data_info.size == args.L

    return train_set, test_set, data_info


def get_data_batch():
    train_set, _, _ = load_dataset()
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True)
    dataiter = iter(train_loader)
    sample, _ = next(dataiter)
    return sample


def logit_transform(x, dequant=True, constraint=0.9, inverse=False):
    """Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(restricted x).

    Args:
        x: input tensor.
        dequant: whether to do dequantization
        constraint: data constraint before logit.
        inverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
    """

    if inverse:
        logit_x = x

        # Log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(log(constraint) - log(1 - constraint))
        ldj = (F.softplus(logit_x) + F.softplus(-logit_x) -
               F.softplus(-pre_logit_scale))
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)

        # Inverse logit transform
        x = 1 / (1 + torch.exp(-logit_x))    # [0.05, 0.95]

        # Unrestrict data
        x *= 2    # [0.1, 1.9]
        x -= 1    # [-0.9, 0.9]
        x /= constraint    # [-1, 1]
        x += 1    # [0, 2]
        x /= 2    # [0, 1]

        return x, ldj

    else:
        if dequant:
            # Dequantization
            noise = torch.rand_like(x)
            x = (x * 255 + noise) / 256

        # Restrict data
        x *= 2    # [0, 2]
        x -= 1    # [-1, 1]
        x *= constraint    # [-0.9, 0.9]
        x += 1    # [0.1, 1.9]
        x /= 2    # [0.05, 0.95]

        # Logit transform
        logit_x = torch.log(x) - torch.log(1 - x)

        # Log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(log(constraint) - log(1 - constraint))
        ldj = (F.softplus(logit_x) + F.softplus(-logit_x) -
               F.softplus(-pre_logit_scale))
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)

        return logit_x, ldj


def log_transform(x, inverse=False, alpha=5.0, epsilon=1e-6):
    """Applies a soft log transformation to map [0,1] into an unbounded space.

    Args:
        x: Input tensor.
        inverse: If True, applies the inverse transformation.
        alpha: Scaling parameter for the transformation.
        epsilon: Small constant to avoid numerical issues.

    Returns:
        Transformed tensor and log-determinant of the Jacobian (LDJ).
    """

    log_alpha = torch.log(torch.tensor(1 + alpha))  # Precompute log(1 + alpha)

    if inverse:
        # Inverse soft log transform: x = (exp(y * log(1 + alpha)) - 1) / alpha
        y = x
        x = (torch.exp(y * log_alpha) - 1) / alpha
        ldj = y * log_alpha  # Log determinant of Jacobian: y * log(1 + alpha)
        return x, ldj

    else:
        # Soft log transform: y = log(1 + alpha * x) / log(1 + alpha)
        x = x + epsilon  # Ensure numerical stability
        y = torch.log(1 + alpha * x) / log_alpha
        ldj = -y * log_alpha  # Log determinant of Jacobian: -y * log(1 + alpha)
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)
        return y, ldj

def gamma_transform(x, dequant=True, constraint=0.9, inverse=False):
    """
    Transforms data from [0, 1] to [0, ∞) using a logit + softplus transform.
    This is appropriate when fitting to a Gamma distribution.

    Returns:
        transformed tensor and log-determinant of Jacobian.
    """
    if inverse:
        # Inverse of softplus is log(exp(x) - 1)
        x_sp = torch.log(torch.exp(x) - 1 + 1e-6)  # small epsilon for numerical stability

        # Inverse of logit transform
        x_scaled = x_sp
        x = torch.sigmoid(x_scaled)

        # Undo constraint
        x = (x * 2 - 1) * constraint
        x = (x + 1) / 2
        return x, None  # You can derive inverse LDJ too if needed

    else:
        if dequant:
            noise = torch.rand_like(x)
            x = (x * 255 + noise) / 256

        # constrain to [0.05, 0.95] or via generic constraint
        x = (x * 2 - 1) * constraint
        x = (x + 1) / 2  # [0.5 - c/2, 0.5 + c/2]
        x_logit = torch.log(x) - torch.log(1 - x)

        # Softplus to ensure positive support
        y = F.softplus(x_logit)

        # LDJ: log |dy/dx| = log(softplus'(logit(x)) * dlogit/dx)
        # softplus'(z) = sigmoid(z)
        # dlogit/dx = 1 / (x(1 - x))
        sigmoid_xlogit = torch.sigmoid(x_logit)
        dlogit_dx = 1 / (x * (1 - x) + 1e-6)
        ldj = torch.log(sigmoid_xlogit * dlogit_dx + 1e-6)
        ldj = ldj.view(ldj.shape[0], -1).sum(dim=1)

        return y, ldj

def exp_forward(u, T=.5, eps=1e-12):
    # z = T·log(1+exp(u/T)) + eps
    z = T * F.softplus(u / T) + eps
    # Jacobian: dz/du = sigmoid(u/T)
    ldj = torch.log(torch.sigmoid(u / T) + 1e-12)  # per‑element
    ldj = ldj.flatten(1).sum(1)
    return z, ldj

def exp_inverse(z, T=.5, eps=1e-12):
    """
    Inverse of z = T * softplus(u/T) + eps.
    Returns:
      u    : tensor such that exp_forward(u) ≈ (z, _)
      ildj : log-det of inverse Jacobian per batch element (shape [B])
    """
    # 1) subtract eps and divide by T → a = (z - eps)/T = softplus(u/T)
    a = (z - eps) / T

    # 2) invert softplus: softplus_inv(a) = log(expm1(a))
    #    clamp to avoid log(0)
    expm1_a = torch.expm1(a).clamp(min=1e-12)
    u = T * torch.log(expm1_a)

    # 3) compute inverse‐Jacobian: du/dz = 1 / (d z/du)
    #    dz/du = T * sigmoid(u/T)  ⇒  du/dz = 1 / (T * sigmoid(u/T))
    # but simpler: from z=T*log(1+e^{u/T})+ε ⇒ du/dz = 1 / (exp(a)-1)/T * exp(a)*T = exp(a)/(exp(a)-1)
    #    so ildj = sum(log exp(a)/(exp(a)-1)) = -sum(log(1 - exp(-a)))
    one_minus_em = (1 - torch.exp(-a)).clamp(min=1e-12)
    ildj = -torch.log(one_minus_em).flatten(1).sum(dim=1)

    return u, ildj