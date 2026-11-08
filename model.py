from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ExperimentConfig:
    code: str
    depth: int
    activation: str
    hidden_init: str
    optimizer: str
    learning_rate: float


CONFIGURATIONS = (
    ExperimentConfig("A", 6, "Sigmoid", "xavier", "Adam", 0.003),
    ExperimentConfig("B", 6, "Tanh", "xavier", "Adam", 0.003),
    ExperimentConfig("C", 6, "ReLU", "xavier", "Adam", 0.003),
    ExperimentConfig("D", 6, "ReLU", "he", "Adam", 0.003),
    ExperimentConfig("E", 6, "ReLU", "he", "SGD", 0.05),
    ExperimentConfig("F", 1, "ReLU", "he", "Adam", 0.003),
)


class MLP(nn.Module):
    def __init__(self, depth: int, activation: str):
        super().__init__()
        activation_types = {"Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ReLU": nn.ReLU}
        if depth < 1:
            raise ValueError("MLP must have at least one hidden layer")
        if activation not in activation_types:
            raise ValueError(f"Unknown activation: {activation}")

        self.hidden_layers = nn.ModuleList([
            nn.Linear(2 if index == 0 else 32, 32, device="cpu", dtype=torch.float32)
            for index in range(depth)
        ])
        self.activation = activation_types[activation]()
        self.output_layer = nn.Linear(32, 2, device="cpu", dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            features = self.activation(layer(features))
        return self.output_layer(features)


def initialize_model(model: MLP, hidden_init: str) -> None:
    if hidden_init not in {"xavier", "he"}:
        raise ValueError(f"Unknown hidden initialization: {hidden_init}")

    for layer in model.modules():
        if not isinstance(layer, nn.Linear):
            continue
        if layer is model.output_layer or hidden_init == "xavier":
            nn.init.xavier_normal_(layer.weight, gain=1.0)
        else:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(layer.bias)


def build_model(configuration: ExperimentConfig, seed: int) -> MLP:
    torch.manual_seed(seed)
    model = MLP(configuration.depth, configuration.activation)
    initialize_model(model, configuration.hidden_init)
    return model


def check_model(
    configuration: ExperimentConfig,
    seed: int,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    model = build_model(configuration, seed)
    repeated_model = build_model(configuration, seed)
    repeated_initialization = all(
        torch.equal(actual, repeated)
        for actual, repeated in zip(model.parameters(), repeated_model.parameters(), strict=True)
    )
    model.zero_grad(set_to_none=True)
    logits = model(features)
    loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
    loss.backward()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = 96 + (configuration.depth - 1) * 1056 + 66
    hidden_layers = [
        layer for layer in model.modules()
        if isinstance(layer, nn.Linear) and layer is not model.output_layer
    ]
    gradients = [layer.weight.grad for layer in hidden_layers]
    checks = {
        "input_shape": features.shape == (8, 2),
        "output_shape": logits.shape == (8, 2),
        "scalar_loss": loss.ndim == 0,
        "finite_loss": torch.isfinite(loss).all().item(),
        "parameter_count": parameter_count == expected_parameter_count,
        "hidden_gradients_present": all(gradient is not None for gradient in gradients),
        "hidden_gradient_shapes": all(
            gradient is not None and gradient.shape == layer.weight.shape
            for layer, gradient in zip(hidden_layers, gradients, strict=True)
        ),
        "hidden_gradients_finite": all(
            gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients
        ),
        "zero_biases": all(
            torch.count_nonzero(layer.bias).item() == 0
            for layer in model.modules() if isinstance(layer, nn.Linear)
        ),
        "repeatable_initialization": repeated_initialization,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(
            f"Model checks failed for {configuration.code}, seed {seed}: " + ", ".join(failed_checks)
        )

    result = {
        "config": configuration.code,
        "seed": seed,
        "depth": configuration.depth,
        "activation": configuration.activation,
        "hidden_init": configuration.hidden_init,
        "input_shape": list(features.shape),
        "output_shape": list(logits.shape),
        "loss_shape": list(loss.shape),
        "check_loss": loss.item(),
        "parameter_count": parameter_count,
        "expected_parameter_count": expected_parameter_count,
        "hidden_gradient_shapes": [
            list(gradient.shape) if gradient is not None else None for gradient in gradients
        ],
        "checks": {name: bool(passed) for name, passed in checks.items()},
    }
    model.zero_grad(set_to_none=True)
    return result
