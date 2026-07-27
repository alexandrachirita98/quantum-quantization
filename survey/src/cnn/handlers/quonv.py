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
"""
import torch
import torch.nn as nn


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
