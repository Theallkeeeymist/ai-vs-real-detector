"""
Training utilities - CORRECTED for binary classification
This is the right way to calculate metrics for your task.
"""

import torch
import torchmetrics
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
import sys


def get_metrics_trackers(device: str):
    """
    Create metric trackers for BINARY classification.
    ⭐ KEY: Use task="binary" not "multiclass" for 2-class problem
    """
    # For binary classification, use task="binary"
    acc_fn = torchmetrics.Accuracy(task="binary").to(device)
    f1_score = torchmetrics.F1Score(task="binary").to(device)
    precision_score = torchmetrics.Precision(task="binary").to(device)
    recall_score = torchmetrics.Recall(task="binary").to(device)
    
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
    Training step with binary classification metrics.
    ⭐ KEY FIXES:
    1. Use task="binary" metrics
    2. Reset at epoch start
    3. Update per batch, compute once at end
    """
    # ⭐ Reset metrics at start of epoch
    accuracy_fn.reset()
    f1_score.reset()
    precision.reset()
    recall.reset()
    
    epoch_loss = 0.0
    batch_count = 0
    
    model.to(device)
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
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

        # Accumulate loss
        epoch_loss += batch_loss.item()
        batch_count += 1
        
        # ⭐ Update metrics with FLOAT predictions for binary task
        # For binary metrics, predictions should be float (probabilities) not int
        y_probs = torch.softmax(y_logits, dim=1)[:, 1]  # Get probability of class 1
        
        accuracy_fn.update(y_preds, y)
        f1_score.update(y_probs, y)
        precision.update(y_probs, y)
        recall.update(y_probs, y)

    # ⭐ Compute metrics once at end of epoch
    train_loss = epoch_loss / batch_count
    train_acc = accuracy_fn.compute().item()
    train_f1 = f1_score.compute().item()
    train_precision = precision.compute().item()
    train_recall = recall.compute().item()

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
    Validation/Test step with binary classification metrics.
    ⭐ Same fixes as train_step
    """
    # ⭐ Reset metrics at start of epoch
    accuracy_fn.reset()
    f1_score.reset()
    precision.reset()
    recall.reset()
    
    epoch_loss = 0.0
    batch_count = 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)

            # Accumulate loss
            batch_loss = loss_fn(y_logits, y)
            epoch_loss += batch_loss.item()
            batch_count += 1
            
            # ⭐ Update metrics with probabilities for binary task
            y_probs = torch.softmax(y_logits, dim=1)[:, 1]  # Get probability of class 1
            
            accuracy_fn.update(y_preds, y)
            f1_score.update(y_probs, y)
            precision.update(y_probs, y)
            recall.update(y_probs, y)

    # ⭐ Compute metrics once at end of epoch
    test_loss = epoch_loss / batch_count
    test_acc = accuracy_fn.compute().item()
    test_f1 = f1_score.compute().item()
    test_precision = precision.compute().item()
    test_recall = recall.compute().item()

    logger.info(f"[TEST]  Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
    logger.info(f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall
    }