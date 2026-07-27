"""Data / training helpers for the `Quantumnet` port (`cnn/models/qtransfer.py`), mirroring the
shipped config in `XanaduAI/quantum-transfer-learning`'s
`c2q_transfer_learning_ants_bees.ipynb`.

- ``download_hymenoptera``: fetches and extracts the torchvision "hymenoptera_data" (ants vs.
  bees) zip the notebook instructs the reader to download by hand.
- ``build_hymenoptera_datasets``: the notebook's `data_transforms` dict (`Resize(256)`,
  `CenterCrop(224)`, `ToTensor`, ImageNet normalization -- augmentation left commented out in the
  original, so not applied here either) plus `datasets.ImageFolder`.
- ``HybridResNet``: `resnet18(pretrained=True)` with every parameter frozen and `.fc` replaced by
  `Quantumnet`, exactly `model_hybrid.fc = Quantumnet()` in the notebook.
- ``train_hybrid``: `train_model`'s train/val loop -- `StepLR` stepped once per epoch, best
  weights (by val accuracy) restored at the end.
- ``RawImageDataset`` / ``HybridLogits``: same adapter pair as `cnn/handlers/qcnn.py`'s
  ``RawImageDataset`` / ``QCNNLogits``, letting `_handlers.evaluation.evaluate_all_metrics` (the
  same all-metrics helper every notebook in this folder uses) run against the hybrid model.
"""
import copy
import os
import zipfile

import requests
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from cnn.models.qtransfer import Quantumnet

HYMENOPTERA_URL = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_hymenoptera(root='./data'):
    """Download and extract hymenoptera_data (ants vs. bees) into `root`, if not already there."""
    target = os.path.join(root, 'hymenoptera_data')
    if os.path.isdir(target):
        return target

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, 'hymenoptera_data.zip')
    with requests.get(HYMENOPTERA_URL, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    os.remove(zip_path)
    return target


def build_hymenoptera_datasets(data_dir):
    """`data_transforms`: deterministic resize/crop/normalize, matching the original (data
    augmentation is present in the source but commented out, so it is not applied here either).
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image_datasets = {
        phase: datasets.ImageFolder(os.path.join(data_dir, phase), transform)
        for phase in ('train', 'val')
    }
    return image_datasets


class HybridResNet(nn.Module):
    """`resnet18(pretrained=True)` with every parameter frozen, `.fc` replaced by `Quantumnet`.

    Matches the notebook's ``for param in model_hybrid.parameters(): param.requires_grad = False``
    followed by ``model_hybrid.fc = Quantumnet()`` -- only the dressed quantum net (`self.fc`) is
    ever trained; the backbone stays a fixed ImageNet feature extractor.
    """

    def __init__(self, **quantumnet_kwargs):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features   # 512 for resnet18
        self.backbone.fc = Quantumnet(in_features=in_features, **quantumnet_kwargs)

    @property
    def fc(self):
        return self.backbone.fc

    def forward(self, x):
        return self.backbone(x)


def train_hybrid(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device,
                 num_epochs):
    """`train_model`: alternating train/val phases per epoch, `scheduler.step()` once per epoch at
    the start of the 'train' phase, restoring the best-val-accuracy weights at the end.
    """
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ('train', 'val'):
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f'[epoch {epoch + 1}/{num_epochs}] {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f'Finished training. Best val accuracy: {best_acc:.4f}')
    return model


class RawImageDataset(Dataset):
    """Wraps an `ImageFolder`'s underlying samples so `evaluate_all_metrics` gets `(image, label)`
    pairs with the same transform used for training/validation -- a thin pass-through, since
    `ImageFolder` already returns exactly that.
    """

    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class HybridLogits(nn.Module):
    """Adapts `HybridResNet` to the `evaluate_all_metrics` contract: images in, logits out.

    `HybridResNet.forward` already returns raw 2-class logits, so this is a direct pass-through --
    kept as a wrapper only so `evaluate_all_metrics` (which does `net(inputs)`) has a name matching
    the pattern the other quantum notebooks use.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)
