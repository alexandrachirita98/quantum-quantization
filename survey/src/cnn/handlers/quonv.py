"""Data / training helpers for the `QNNModel` port (`cnn/models/quonv.py`), mirroring the shipped
config in `PlanQK/variational-quanvolutional-neural-networks`'s `generate_experiments.py` /
`learner.py` / `runners.py`.

- ``freeze_untrainable``: `learner.py::training_experiment`'s parameter-freeze — when the run is
  configured `trainable=False`, the quantum layer's `weights` never receives gradient updates (the
  original additionally rewrites the circuit to ignore whatever gradient *would* have been applied;
  `QuonvLayer` here sidesteps that by registering `weights` as a plain buffer instead of a
  `Parameter` in that mode, so this is a no-op safety check, not the actual freeze).
- ``train_quonv``: `Learner.train`'s loop — plain mini-batch SGD via Adam and cross-entropy,
  capped at `steps_in_epoch` steps per epoch like the original's early `break`.
- ``RawImageDataset`` / ``QuonvLogits``: same adapter pair as `cnn/handlers/qcnn.py`'s
  ``RawImageDataset`` / ``QCNNLogits``, letting `_handlers.evaluation.evaluate_all_metrics` (the
  same all-metrics helper every notebook in this folder uses) run against ``QNNModel``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def freeze_untrainable(model):
    """Assert the quantum layer's weights are not a trainable Parameter when `trainable=False`,
    matching `learner.py`'s `requires_grad = False` gate on `qlayer_1.torch_qlayer.weights`.
    """
    if not model.qlayer.trainable:
        assert not isinstance(model.qlayer.weights, nn.Parameter)


def train_quonv(model, optimizer, criterion, train_loader, test_loader, device,
                epochs, steps_in_epoch=100, log_every=10):
    """`Learner.train`/`Learner.validate`: one pass per epoch over `train_loader`, capped at
    `steps_in_epoch` batches (the original's `if step % (steps_in_epoch - 1) == 0: break`), then a
    validation pass over `test_loader` at the end of each epoch.
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            if step >= steps_in_epoch:
                break
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


def resize_images(images, img_size):
    """images: (B, H, W) or (B, 1, H, W) uint8/float, any range -> (B, 1, img_size, img_size) in
    [0, 1] float, matching `QNNModel`'s expected input (`Threshold_Encoder`'s `threshold=0.5` is
    calibrated against raw `[0, 1]` pixel values, same as `generate_experiments.py`'s un-normalized
    default transform).
    """
    x = images.reshape(images.size(0), 1, images.size(-2), images.size(-1)).float() / 255.
    return F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)


class RawImageDataset(Dataset):
    """Wraps a medmnist ``(imgs, labels)`` numpy pair as a plain ``Dataset`` of raw 28x28 images.

    Used to feed ``_handlers.evaluation.evaluate_all_metrics`` — it expects a ``Dataset`` of
    ``(image, label)`` pairs, unlike ``train_quonv`` above which trains on already-resized
    ``img_size x img_size`` tensors. Images stay uint8-range floats (0-255); ``QuonvLogits`` below
    resizes and normalizes each batch, exactly like the training path.
    """

    def __init__(self, imgs, labels):
        self.imgs = torch.as_tensor(imgs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels).reshape(-1).long()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


class QuonvLogits(nn.Module):
    """Adapts ``QNNModel`` to the ``evaluate_all_metrics`` contract: raw images in, logits out.

    ``evaluate_all_metrics`` calls ``net(inputs).softmax(dim=-1)`` to recover class
    probabilities; ``QNNModel`` already returns raw linear-head logits, so this only needs to
    resize the native-resolution images down to the model's ``img_size`` first.
    """

    def __init__(self, model, img_size):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, images):
        device = next(self.model.parameters()).device
        x = resize_images(images, self.img_size).to(device)
        return self.model(x).float()
