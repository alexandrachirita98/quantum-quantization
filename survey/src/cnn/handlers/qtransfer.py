"""Data / training helpers for the `Quantumnet` port (`cnn/models/qtransfer.py`), mirroring the
shipped config in `XanaduAI/quantum-transfer-learning`'s
`c2q_transfer_learning_ants_bees.ipynb`, but trained on **BreastMNIST** instead of the original's
ants-vs-bees `ImageFolder` set, matching every other notebook in this folder
(`qcnn.ipynb`, `quonv.ipynb`).

- ``build_breastmnist_datasets``: BreastMNIST's native 28x28 grayscale images, replicated to 3
  channels and resized/normalized the same way the original's `data_transforms` prepares its RGB
  ants/bees images (`Resize(256)`, `CenterCrop(224)`, ImageNet normalization) -- required because
  `resnet18(pretrained=True)` expects 3-channel ImageNet-statistics input regardless of the
  downstream task.
- ``HybridResNet``: `resnet18(pretrained=True)` with every parameter frozen and `.fc` replaced by
  `Quantumnet`, exactly `model_hybrid.fc = Quantumnet()` in the notebook.
- ``train_hybrid``: `train_model`'s train/val loop -- `StepLR` stepped once per epoch, best
  weights (by val accuracy) restored at the end.
- ``HybridLogits``: same adapter role as `cnn/handlers/qcnn.py`'s ``QCNNLogits``, letting
  `_handlers.evaluation.evaluate_all_metrics` (the same all-metrics helper every notebook in this
  folder uses) run against the hybrid model.
"""
import copy

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from cnn.models.qtransfer import Quantumnet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_rgb_uint8(images):
    """images: (B, 28, 28) or (B, 1, 28, 28) uint8/float -> (B, 1, 28, 28) float in [0, 1]."""
    return images.reshape(images.size(0), 1, 28, 28).float() / 255.


class BreastMNISTRGBDataset(Dataset):
    """Wraps a medmnist BreastMNIST ``(imgs, labels)`` numpy pair, replicating the single
    grayscale channel to 3 and applying the original's `data_transforms` (`Resize(256)`,
    `CenterCrop(224)`, ImageNet normalization) so a pretrained `resnet18` can consume it, matching
    the shapes/statistics its ImageNet weights expect.
    """

    def __init__(self, imgs, labels):
        self.imgs = torch.as_tensor(imgs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels).reshape(-1).long()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        img = _to_rgb_uint8(self.imgs[index].unsqueeze(0)).squeeze(0).expand(3, 28, 28)
        return self.transform(img), self.labels[index]


def build_breastmnist_datasets(root='./data'):
    """Downloads BreastMNIST (via medmnist) and wraps each split as a `BreastMNISTRGBDataset`."""
    import os

    from medmnist import BreastMNIST

    os.makedirs(root, exist_ok=True)   # BreastMNIST() checks root exists *before* downloading
    raw = {phase: BreastMNIST(split=phase, download=True, root=root) for phase in ('train', 'val')}
    return {phase: BreastMNISTRGBDataset(raw[phase].imgs, raw[phase].labels) for phase in raw}


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
