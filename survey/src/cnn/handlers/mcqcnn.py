"""Data / training helpers for the MC-QCNN port (`cnn/models/mcqcnn.py`), mirroring the shipped
config in `anthonysmaldone/QCNN-Multi-Channel-Supervised-Learning`'s `train.py` / `prepare_data.py`.

- ``resize_images``: the original's `prepare_data.py` resize to the `10x10` inputs every model in
  `models.py` declares (`tf.keras.layers.Input((10,10,3))`), plus the grayscale->3-channel
  expansion described below. Output is **channel-last** `(B, H, W, C)`, matching the TF layout
  `MCQCNNModel.forward` expects, not PyTorch's usual `(B, C, H, W)`.
- ``train_mcqcnn``: `train.py::train_model`'s `model.fit` -- plain mini-batch Adam over
  `sparse_categorical_crossentropy` (`nn.CrossEntropyLoss` here, since the port leaves the softmax
  folded into the loss), validating on the test split once per epoch.
- ``RawImageDataset`` / ``MCQCNNLogits``: same adapter pair as `cnn/handlers/quonv.py`'s
  ``RawImageDataset`` / ``QuonvLogits``, letting `_handlers.evaluation.evaluate_all_metrics` (the
  same all-metrics helper every notebook in this folder uses) run against ``MCQCNNModel``.

**On the channel count.** The paper's whole subject is *multi-channel* data, and every variant in
`models.py` is built for a 3- or 12-channel input; the ten variants only differ from each other in
how they combine channels, so a 1-channel input would collapse CO / PCO / PCO-T / WEV / control
into near-identical models. BreastMNIST is grayscale, so ``resize_images`` replicates the single
channel into ``n_channels`` -- the same grayscale->RGB adaptation `cnn/handlers/qtransfer.py` makes
for ResNet18, and it maps onto the original's 3-channel `COLORS`/`CIFAR10` path (`registers=3`).
Note this means the three channels carry identical data: the multi-channel *machinery* is
exercised (3 registers, inter-register entanglement, per-channel parameter sets), but the
inter-channel *information* the paper's ansatzes are designed to capture is not there to find.
Point ``args.dataset`` at a natively-RGB MedMNIST flag (``bloodmnist``, ``pathmnist``,
``dermamnist``, ``retinamnist``) for a run where the channels actually differ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def resize_images(images, img_size=10, n_channels=3):
    """images: (B, H, W), (B, H, W, C) or (B, 1, H, W), uint8-range -> (B, img_size, img_size,
    n_channels) float in [0, 1], channel-last.

    Bilinear down-sampling to the `10x10` inputs `models.py` hardcodes. A single-channel source is
    replicated across `n_channels`; a source that already has `n_channels` is passed through.
    """
    x = torch.as_tensor(images).float()
    if x.dim() == 3:                                  # (B, H, W) grayscale
        x = x.unsqueeze(1)
    elif x.dim() == 4 and x.shape[-1] in (1, 3):      # (B, H, W, C) channel-last medmnist RGB
        x = x.permute(0, 3, 1, 2)
    # x is now (B, C, H, W)
    x = x / 255.
    x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)

    if x.size(1) == 1 and n_channels > 1:
        x = x.expand(-1, n_channels, -1, -1)
    if x.size(1) != n_channels:
        raise ValueError(f'expected {n_channels} channels, got {x.size(1)}')

    return x.permute(0, 2, 3, 1).contiguous()         # -> (B, H, W, C), the TF layout


def train_mcqcnn(model, optimizer, criterion, train_loader, test_loader, device,
                 epochs, log_every=5):
    """`train.py::train_model`'s `model.fit(...)`: `epochs` passes over the training set with
    mini-batch Adam, reporting validation accuracy at the end of each epoch.
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % log_every == 0:
                print(f'epoch {epoch}/{epochs}  step {step + 1}  loss: {loss.item():.4f}')

        val_acc = evaluate(model, test_loader, device)
        print(f'[epoch {epoch}] train_loss: {running_loss / (step + 1):.4f}  val_acc: {val_acc:.4f}')


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


class RawImageDataset(Dataset):
    """Wraps a medmnist ``(imgs, labels)`` numpy pair as a plain ``Dataset`` of raw 28x28 images.

    Used to feed ``_handlers.evaluation.evaluate_all_metrics`` -- it expects a ``Dataset`` of
    ``(image, label)`` pairs, unlike ``train_mcqcnn`` above which trains on already-resized
    ``img_size x img_size`` tensors. Images stay uint8-range floats (0-255); ``MCQCNNLogits``
    below resizes and normalizes each batch, exactly like the training path.
    """

    def __init__(self, imgs, labels):
        self.imgs = torch.as_tensor(imgs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels).reshape(-1).long()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


class MCQCNNLogits(nn.Module):
    """Adapts ``MCQCNNModel`` to the ``evaluate_all_metrics`` contract: raw images in, logits out.

    ``evaluate_all_metrics`` calls ``net(inputs).softmax(dim=-1)`` to recover class probabilities;
    ``MCQCNNModel`` already returns raw linear-head logits, so this only needs to apply the same
    resize + channel expansion the training path uses.
    """

    def __init__(self, model, img_size=10, n_channels=3):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.n_channels = n_channels

    def forward(self, images):
        device = next(self.model.parameters()).device
        x = resize_images(images, self.img_size, self.n_channels).to(device)
        return self.model(x).float()
