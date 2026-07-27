"""Reusable training / evaluation helpers for the MedViTV2 pipeline.

Extracted from ``notebooks/classical/Evaluation.ipynb`` (which mirrors MedViTV2's ``main.py``).
The notebook keeps only the configuration and the linear pipeline; every reusable piece —
checkpoint download, model selection, the two training routines, the metric helpers and the
all-metrics evaluation — lives here.
"""
import os
import sys

import numpy as np
import requests
import torch
import torch.utils.data as data
from tqdm import tqdm
from medmnist import Evaluator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
MODEL_URLS = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0",
}


def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")


def build_model(model_name, nb_classes, model_classes, pretrained=False, checkpoint_path=None):
    """Instantiate a MedViT_* model or fall back to any timm model (CNN or transformer).

    ``model_classes`` maps names to MedViT builder classes; it is passed in so this module does
    not need to import the cloned MedViTV2 repo. Any name that is not a key falls through to
    ``timm.create_model``.
    """
    if model_name in model_classes:
        net = model_classes[model_name](num_classes=nb_classes).cuda()
        if pretrained:
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                url = MODEL_URLS.get(model_name)
                if not url:
                    raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                checkpoint_path = f'./{model_name}.pth'
                download_checkpoint(url, checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        import timm
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()
    return net


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity


def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()    # Sum of all elements in the matrix
    return tp_tn_sum / total_sum


# --------------------------------------------------------------------------- #
# Training routines
# --------------------------------------------------------------------------- #
def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task):
    """MedMNIST training loop: validates each epoch with the medmnist Evaluator (AUC/ACC)."""
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)

                y_score = torch.cat((y_score, outputs.cpu()), 0)

        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
        metrics = evaluator.evaluate(y_score)

        val_accurate, _ = metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')


def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path):
    """Image-folder training loop: validates each epoch with the full scikit-learn metric suite."""
    best_acc = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # Training Loop
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # Validation Loop
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store raw probabilities/logits for AUC
        acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # Raw outputs (logits)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities

                predict_y = torch.max(probs, dim=1)[1]  # Predicted class

                # Collect predictions, labels, and probabilities
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Calculate accuracy
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # Calculate metrics
        val_accurate = acc / len(test_loader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')  # Sensitivity
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Confusion Matrix for multi-class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)  # List of specificities per class
        avg_specificity = sum(specificity) / len(specificity)  # Average specificity

        # Overall Accuracy calculation
        overall_acc = overall_accuracy(conf_matrix)

        # One-hot encode the labels for AUC calculation
        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            # Compute AUC for multi-class
            auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')  # Handle edge case where AUC can't be computed

        # Print metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} '
              f'val_accuracy: {val_accurate:.4f} precision: {precision:.4f} '
              f'recall: {recall:.4f} specificity: {avg_specificity:.4f} '
              f'f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}')

        # Save best model
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')


# --------------------------------------------------------------------------- #
# Full evaluation (all metrics on one split)
# --------------------------------------------------------------------------- #
def evaluate_all_metrics(net, dataset, dataset_name, nb_classes, device, split='test', batch_size=48, size=224):
    """Run ``net`` once over ``dataset`` and report every metric the two routines can produce.

    Prints the medmnist Evaluator AUC/ACC (for ``*mnist`` datasets) plus the full scikit-learn
    suite, and returns everything as a dict. The loader is unshuffled so predictions stay in
    split order (required by the medmnist Evaluator).

    ``size`` selects which medmnist ``.npz`` resolution the Evaluator reads (``224`` for the timm
    pipelines here, ``28`` — the native resolution — for e.g. the QCNN amplitude-encoding pipeline).
    """
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    net.eval()
    y_score = torch.tensor([])            # predicted class probabilities, in split order
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, file=sys.stdout):
            outputs = net(inputs.to(device)).softmax(dim=-1)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            labels = targets.squeeze().long().cpu().numpy()
            all_labels.extend(np.atleast_1d(labels).tolist())
    y_score = y_score.numpy()

    results = {}

    # Official medmnist metrics (only for the *mnist datasets)
    if dataset_name.endswith('mnist'):
        auc_mm, acc_mm = Evaluator(dataset_name, split, size=size, root='./data').evaluate(y_score)
        results['medmnist_auc'], results['medmnist_acc'] = auc_mm, acc_mm
        print(f'[medmnist Evaluator]  auc: {auc_mm:.4f}  acc: {acc_mm:.4f}\n')

    # General classification metrics (any dataset)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(nb_classes)))
    accuracy    = accuracy_score(all_labels, all_preds)
    ov_acc      = overall_accuracy(conf_matrix)
    precision   = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall      = recall_score(all_labels, all_preds, average='weighted', zero_division=0)  # sensitivity
    f1          = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    spec_list   = specificity_per_class(conf_matrix)
    specificity = sum(spec_list) / len(spec_list)

    try:
        if nb_classes == 2:
            auc = roc_auc_score(all_labels, y_score[:, 1])
        else:
            labels_1hot = label_binarize(all_labels, classes=list(range(nb_classes)))
            auc = roc_auc_score(labels_1hot, y_score[:, :nb_classes], multi_class='ovr')
    except ValueError:
        auc = float('nan')

    results.update(accuracy=accuracy, overall_accuracy=ov_acc, auc=auc, precision=precision,
                   recall=recall, specificity=specificity, f1=f1,
                   specificity_per_class=spec_list, confusion_matrix=conf_matrix)

    print(f'=== {split} metrics ({nb_classes} classes) ===')
    print(f'accuracy            : {accuracy:.4f}')
    print(f'overall_accuracy    : {ov_acc:.4f}')
    print(f'auc (ovr)           : {auc:.4f}')
    print(f'precision (weighted): {precision:.4f}')
    print(f'recall / sensitivity: {recall:.4f}')
    print(f'specificity (avg)   : {specificity:.4f}')
    print(f'f1 (weighted)       : {f1:.4f}')
    print('\nper-class specificity:', [f'{s:.3f}' for s in spec_list])
    print('\nconfusion matrix:')
    print(conf_matrix)
    print('\nclassification report:')
    print(classification_report(all_labels, all_preds, zero_division=0))
    return results
