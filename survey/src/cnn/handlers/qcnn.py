"""Data / training helpers for the `QCNN` port (`cnn/models/qcnn.py`), mirroring the shipped
config in `takh04/QCNN`'s `result.py` and `Training.py`.

- ``to_amplitude_states``: the original's ``'resize256'`` feature-reduction mode —
  ``tf.image.resize(X, (256, 1))`` (bilinear) then L2-normalize, feeding `AmplitudeEmbedding`'s
  256 = 2^8 amplitudes. Ported here as `F.interpolate(..., mode='bilinear')` on the flattened
  pixel column, which is the same operation torch-side.
- ``cross_entropy_loss``: `Training.py::cross_entropy`, batched instead of the original's
  Python-loop-over-samples sum.
- ``predict`` / ``accuracy``: `Benchmarking.py::accuracy_test`'s decoding rule, `P=0` if
  `p[0] > p[1]` else `P=1`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def to_amplitude_states(images):
    """images: (B, 28, 28) or (B, 1, 28, 28) uint8/float, any range -> (B, 256) L2-normalized.

    Matches `data.py`'s `'resize256'` mode: flatten to a (784, 1) column, bilinear-resize to
    (256, 1), squeeze back to a vector, L2-normalize (`AmplitudeEmbedding(..., normalize=True)`).
    """
    x = images.reshape(images.size(0), 1, -1, 1).double() / 255.
    x = F.interpolate(x, size=(256, 1), mode='bilinear', align_corners=False)
    x = x.reshape(images.size(0), 256)
    x = x + 1e-12  # avoid an exact all-zero (unnormalizable) state
    return x / x.norm(dim=1, keepdim=True)


def cross_entropy_loss(probs, labels):
    """probs: (B, 2) = [P(0), P(1)] from `QCNN.forward`; labels: (B,) integer 0/1.

    `Training.py::cross_entropy`: -sum(l*log(p[l]) + (1-l)*log(1-p[1-l])) / B, i.e. plain binary
    cross-entropy read directly off the two class probabilities (already normalized, so this is
    ``F.nll_loss`` on ``probs.log()`` rather than ``F.cross_entropy``, which would apply its own
    softmax on top of an already-normalized distribution).
    """
    probs = probs.clamp(min=1e-12, max=1.0)
    return F.nll_loss(probs.log(), labels)


def predict(probs):
    """`Benchmarking.py::accuracy_test`: predicted class is 0 if P(0) > P(1) else 1."""
    return (probs[:, 0] <= probs[:, 1]).long()


def accuracy(probs, labels):
    return (predict(probs) == labels).float().mean().item()


def train_qcnn(model, optimizer, train_x, train_y, test_x, test_y, steps, batch_size, device,
               log_every=10):
    """`Training.py::circuit_training`: minibatches sampled with replacement each step (not
    epochs over a shuffled loader, matching the original), optimizing the batched cross-entropy.
    Reports test accuracy after training, via `predict`/`accuracy` above.
    """
    train_x, train_y = train_x.to(device), train_y.to(device)
    n = train_x.size(0)

    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        batch_x, batch_y = train_x[idx], train_y[idx]

        probs = model(batch_x)
        loss = cross_entropy_loss(probs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            acc = accuracy(probs, batch_y)
            print(f'step {step + 1}/{steps}  loss: {loss.item():.4f}  batch_acc: {acc:.4f}')

    model.eval()
    with torch.no_grad():
        test_probs = model(test_x.to(device))
        test_acc = accuracy(test_probs, test_y.to(device))
    print(f'\nFinished training. test_accuracy: {test_acc:.4f}')
    return test_acc


class RawImageDataset(Dataset):
    """Wraps a medmnist ``(imgs, labels)`` numpy pair as a plain ``Dataset`` of raw 28x28 images.

    Used to feed ``_handlers.evaluation.evaluate_all_metrics`` — it expects a ``Dataset`` of
    ``(image, label)`` pairs, unlike ``train_qcnn`` above which trains on the pre-amplitude-encoded
    tensors directly. Images stay uint8-range floats (0-255); ``QCNNLogits`` below runs
    ``to_amplitude_states`` per batch, exactly like the training path.
    """

    def __init__(self, imgs, labels):
        self.imgs = torch.as_tensor(imgs, dtype=torch.float64)
        self.labels = torch.as_tensor(labels).reshape(-1).long()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


class QCNNLogits(nn.Module):
    """Adapts ``QCNN`` to the ``evaluate_all_metrics`` contract: raw images in, logits out.

    ``evaluate_all_metrics`` calls ``net(inputs).softmax(dim=-1)`` to recover class
    probabilities. ``QCNN.forward`` already returns normalized ``[P(0), P(1)]`` marginals, so
    this wraps it as ``to_amplitude_states -> QCNN -> log(probs)``: softmax undoes the log and
    reproduces the same probabilities exactly, letting the shared evaluation helper run unchanged.
    """

    def __init__(self, qcnn):
        super().__init__()
        self.qcnn = qcnn

    def forward(self, images):
        states = to_amplitude_states(images).to(next(self.qcnn.parameters()).device)
        probs = self.qcnn(states)
        return probs.clamp(min=1e-12, max=1.0).log().float()
