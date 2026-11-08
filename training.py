from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from data import DataSplit, PreparedData
from model import ExperimentConfig, MLP, build_model


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


@dataclass
class ExperimentResult:
    model: MLP
    metrics: list[dict]
    gradients: list[dict]


def create_train_loader(train: DataSplit, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(10000 + seed)
    return DataLoader(
        TensorDataset(train.features, train.labels),
        batch_size=128,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def make_optimizer(model: nn.Module, configuration: ExperimentConfig) -> Optimizer:
    if configuration.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    if configuration.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=configuration.learning_rate, momentum=0)
    raise ValueError(f"Unknown optimizer: {configuration.optimizer}")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise ValueError("Training loss contains NaN or Inf; inspect the batch and forward pass")
        loss.backward()
        optimizer.step()


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 128,
) -> Metrics:
    if batch_size < 1 or len(labels) == 0 or len(features) != len(labels):
        raise ValueError("Evaluation requires a positive batch size and matching nonempty tensors")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    total_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch_features = features[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Evaluation loss contains NaN or Inf; inspect the batch and forward pass")
            total_loss += loss.item() * len(batch_labels)
            correct_predictions += int((logits.argmax(dim=1) == batch_labels).sum().item())
    return Metrics(loss=total_loss / len(labels), accuracy=correct_predictions / len(labels))


def measure_gradients(model: MLP, features: torch.Tensor, labels: torch.Tensor) -> list[float]:
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        if not torch.isfinite(features).all().item():
            raise ValueError("Probe features contain NaN or Inf")
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="mean")(model(features), labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Probe loss contains NaN or Inf; inspect the batch and forward pass")
            loss.backward()
        hidden_layers = [
            layer for layer in model.modules()
            if isinstance(layer, nn.Linear) and layer is not model.output_layer
        ]
        norms = []
        for index, layer in enumerate(hidden_layers, start=1):
            gradient = layer.weight.grad
            if gradient is None or gradient.shape != layer.weight.shape:
                raise ValueError(f"Missing or invalid weight gradient in hidden layer {index}")
            if not torch.isfinite(gradient).all().item():
                raise ValueError(f"Hidden layer {index} gradient contains NaN or Inf; inspect backward")
            norm = float(torch.linalg.matrix_norm(gradient, ord="fro").item())
            if not math.isfinite(norm):
                raise ValueError(f"Hidden layer {index} gradient norm contains NaN or Inf")
            norms.append(norm)
        return norms
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def run_experiment(configuration: ExperimentConfig, seed: int, data: PreparedData) -> ExperimentResult:
    torch.set_num_threads(1)
    model = build_model(configuration, seed)
    optimizer = make_optimizer(model, configuration)
    loader = create_train_loader(data.train, seed)
    metrics = []
    gradients = []
    for epoch in range(101):
        if epoch > 0:
            train_epoch(model, loader, optimizer)
        train_metrics = evaluate(model, data.train.features, data.train.labels)
        validation_metrics = evaluate(model, data.validation.features, data.validation.labels)
        metrics.append({
            "config": configuration.code,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": validation_metrics.loss,
            "val_accuracy": validation_metrics.accuracy,
        })
        if configuration.code in {"A", "C", "D"} and epoch in {0, 20, 100}:
            norms = measure_gradients(model, data.probe.features, data.probe.labels)
            gradients.extend(
                {
                    "config": configuration.code,
                    "seed": seed,
                    "epoch": epoch,
                    "layer": index,
                    "gradient_norm": norm,
                }
                for index, norm in enumerate(norms, start=1)
            )
    model.zero_grad(set_to_none=True)
    return ExperimentResult(model=model, metrics=metrics, gradients=gradients)
