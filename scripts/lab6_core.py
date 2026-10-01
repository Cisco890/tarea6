"""Funciones compartidas del Laboratorio 6 — Transfer Learning (CIFAR-10)."""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import models
from torchvision.models import VGG16_Weights


SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VGG_SIZE = 112
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _vgg_block_slices() -> dict[int, slice]:
    # features indices for MaxPool separators in torchvision VGG-16
    # blocks: 0-4, 5-9, 10-16, 17-23, 24-30
    return {
        1: slice(0, 5),
        2: slice(5, 10),
        3: slice(10, 17),
        4: slice(17, 24),
        5: slice(24, 31),
    }


def build_vgg_feature_extractor(num_classes: int = 10) -> nn.Module:
    weights = VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_vgg_finetune(unfrozen_blocks: list[int], num_classes: int = 10) -> nn.Module:
    weights = VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    slices = _vgg_block_slices()
    for b in unfrozen_blocks:
        for p in model.features[slices[b]].parameters():
            p.requires_grad = True
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def split_vgg_param_groups(model: nn.Module, backbone_lr: float, classifier_lr: float, weight_decay: float):
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    groups.append({"params": classifier_params, "lr": classifier_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    n = 0
    preds_all, labels_all = [], []
    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs
        preds_all.append(outputs.argmax(dim=1).detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n = 0
    preds_all, labels_all = [], []
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs
        preds_all.append(outputs.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(n, 1)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int = 8,
    patience: int = 3,
    experiment_name: str = "run",
    save_dir: str = "outputs_lab6",
):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    cumulative = 0.0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_params, trainable_params = count_parameters(model)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_m = evaluate(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        cumulative += epoch_time

        row = {
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "epoch_time": epoch_time,
            "cumulative_time": cumulative,
        }
        history.append(row)

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(best_state, os.path.join(save_dir, f"{experiment_name}_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    summary = {
        "experiment": experiment_name,
        "epochs_executed": len(history),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_val_accuracy": history[best_epoch - 1]["val_accuracy"],
        "best_val_precision": history[best_epoch - 1]["val_precision"],
        "best_val_recall": history[best_epoch - 1]["val_recall"],
        "best_val_loss": history[best_epoch - 1]["val_loss"],
        "avg_epoch_time_s": float(np.mean([h["epoch_time"] for h in history])),
        "total_training_time_s": cumulative,
        "peak_gpu_memory_mb": peak_mem if peak_mem is not None else np.nan,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history,
    }
    return model, summary


def measure_flops(model, input_size: int, device) -> float:
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, dummy).total()
        return float(flops)
    except Exception:
        try:
            from thop import profile

            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            return float(flops)
        except Exception:
            # fallback rough estimate
            return float("nan")


def measure_latency_ms(model, input_size: int, device, warmup: int = 20, runs: int = 100) -> float:
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)


def confusion_from_preds(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
