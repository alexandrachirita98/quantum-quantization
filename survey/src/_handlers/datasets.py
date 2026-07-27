"""MedMNIST dataset + transform builders for the evaluation pipeline.

Ported from MedViTV2's ``datasets.py`` so a notebook can build the same datasets without cloning
the repository. Only the ``*mnist`` branch is kept: the paper's image-folder datasets (Kvasir,
CPN, Fetal, PAD, ISIC2018) come with ~350 lines of bespoke downloaders upstream and are **not**
ported here — ``build_dataset`` raises ``NotImplementedError`` for them.

The transforms and the ``DataClass(...)`` calls are exactly what upstream uses, so the datasets
returned here are identical to the ones the cloned repo produces.
"""
import os

import medmnist
import torch
from medmnist import INFO
from torchvision import transforms

# upstream seeds at import time; kept so shuffling / augmentation stay reproducible across runs
SEED = 42
torch.manual_seed(SEED)


def build_transform(args=None):
    """Train / test transform pair. ``args`` is accepted (and ignored) to match upstream."""
    t_train = [
        transforms.RandomResizedCrop(224),
        transforms.AugMix(alpha=0.4),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ]
    t_test = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ]
    return transforms.Compose(t_train), transforms.Compose(t_test)


def build_dataset(args, root='./data'):
    """Download (if needed) and return ``(train_dataset, test_dataset, nb_classes)``.

    ``args.dataset`` must be a MedMNIST flag (``breastmnist``, ``pathmnist``, ``bloodmnist``, …).
    Images are served as 3-channel 224x224 (``as_rgb=True``, ``size=224``) and memory-mapped, so
    the first call downloads the ``.npz`` into ``root`` and later calls reuse it.

    ``root`` must match the root used by the medmnist ``Evaluator`` in ``evaluation.py`` (``./data``),
    otherwise the evaluator will not find the split it scores against.
    """
    if not args.dataset.endswith('mnist'):
        raise NotImplementedError(
            f"{args.dataset!r} is not a MedMNIST flag. Only the *mnist datasets are ported here; "
            "the image-folder datasets (Kvasir, CPN, Fetal, PAD, ISIC2018) need MedViTV2's own "
            "downloaders from the upstream datasets.py."
        )

    os.makedirs(root, exist_ok=True)
    train_transform, test_transform = build_transform(args)

    info = INFO[args.dataset]
    n_channels = info['n_channels']
    nb_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    print("Number of channels: ", n_channels)
    print("Number of classes: ", nb_classes)

    train_dataset = DataClass(split='train', transform=train_transform, download=True,
                              as_rgb=True, root=root, size=224, mmap_mode='r')
    test_dataset = DataClass(split='test', transform=test_transform, download=True,
                             as_rgb=True, root=root, size=224, mmap_mode='r')
    return train_dataset, test_dataset, nb_classes
