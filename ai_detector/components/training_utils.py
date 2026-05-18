"""
Training utilities - exact functions from your notebook.
train_step and test_step for each model.
"""

import torch
import torchmetrics
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
import sys


def get_metrics_trackers(device: str):
    """
    Create metric trackers (from your notebook).
    """
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2).to(device)
    precision_score = torchmetrics.Precision(task="multiclass", num_classes=2).to(device)
    recall_score = torchmetrics.Recall(task="multiclass", num_classes=2).to(device)
    
    return acc_fn, f1_score, precision_score, recall_score


def train_step(
    model,
    data_loader,
    accuracy_fn,
    f1_score,
    precision,
    recall,
    loss_fn,
    optimizer,
    device: str = "cuda"
):
    """
    Training step 
    """
    train_acc, train_loss, train_f1, train_precision, train_recall = 0, 0, 0, 0, 0
    
    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)
        
        # Calculate loss
        batch_loss = loss_fn(y_logits, y)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backward pass
        batch_loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate values
        train_loss += batch_loss.item()
        train_acc += accuracy_fn(y_preds, y).item()
        train_f1 += f1_score(y_preds, y).item()
        train_precision += precision(y_preds, y).item()
        train_recall += recall(y_preds, y).item()

    # Average metrics
    n = len(data_loader)
    train_loss /= n
    train_acc /= n
    train_f1 /= n
    train_precision /= n
    train_recall /= n

    logger.info(f"[TRAIN] Accuracy: {train_acc:.4f} | Loss: {train_loss:.4f}")
    logger.info(f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
    
    return {
        "loss": train_loss,
        "accuracy": train_acc,
        "f1": train_f1,
        "precision": train_precision,
        "recall": train_recall
    }


def test_step(
    model,
    data_loader,
    accuracy_fn,
    f1_score,
    precision,
    recall,
    loss_fn,
    device: str = "cuda"
):
    """
    Validation/Test step - EXACT from your notebook.
    """
    test_acc, test_loss, test_f1, test_precision, test_recall = 0, 0, 0, 0, 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)

            # Accumulate values
            test_loss += loss_fn(y_logits, y).item()
            test_acc += accuracy_fn(y_preds, y).item()
            test_f1 += f1_score(y_preds, y).item()
            test_precision += precision(y_preds, y).item()
            test_recall += recall(y_preds, y).item()

    # Average metrics
    n = len(data_loader)
    test_acc /= n
    test_loss /= n
    test_f1 /= n
    test_precision /= n
    test_recall /= n

    logger.info(f"[TEST]  Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
    logger.info(f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall
    }