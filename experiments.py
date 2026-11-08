from dataclasses import dataclass
import math

from data import PreparedData
from model import CONFIGURATIONS, MLP
from training import evaluate, run_experiment


SEEDS = (11, 29)
METRIC_FIELDS = (
    "config", "seed", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
)
GRADIENT_FIELDS = ("config", "seed", "epoch", "layer", "gradient_norm")


@dataclass
class ExperimentSuite:
    metrics: list[dict]
    gradients: list[dict]
    final_metrics: list[dict]
    selection: dict
    selected_model: MLP


def validate_records(rows: list[dict], fields: tuple[str, ...], key_fields: tuple[str, ...], expected_keys: set[tuple]) -> None:
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Results must contain exactly these fields: {', '.join(fields)}")
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError(f"Missing, duplicate or unexpected result keys: {', '.join(key_fields)}")


def validate_results(metrics: list[dict], gradients: list[dict]) -> None:
    expected_metrics = {
        (configuration.code, seed, epoch)
        for configuration in CONFIGURATIONS
        for seed in SEEDS
        for epoch in range(101)
    }
    expected_gradients = {
        (code, seed, epoch, layer)
        for code in ("A", "C", "D")
        for seed in SEEDS
        for epoch in (0, 20, 100)
        for layer in range(1, 7)
    }
    validate_records(metrics, METRIC_FIELDS, ("config", "seed", "epoch"), expected_metrics)
    validate_records(gradients, GRADIENT_FIELDS, ("config", "seed", "epoch", "layer"), expected_gradients)
    for row in metrics:
        if any(not math.isfinite(row[field]) for field in METRIC_FIELDS[3:]):
            raise ValueError(f"Nonfinite metrics for {row['config']}, seed {row['seed']}, epoch {row['epoch']}")
        if any(not 0 <= row[field] <= 1 for field in ("train_accuracy", "val_accuracy")):
            raise ValueError("Accuracy must be a fraction between 0 and 1")
    for row in gradients:
        norm = row["gradient_norm"]
        if not math.isfinite(norm) or norm < 0:
            raise ValueError(f"Invalid gradient norm for {row['config']}, seed {row['seed']}, epoch {row['epoch']}, layer {row['layer']}")


def select_run(final_rows: list[dict]) -> dict:
    expected_keys = {(configuration.code, seed, 100) for configuration in CONFIGURATIONS for seed in SEEDS}
    validate_records(final_rows, METRIC_FIELDS, ("config", "seed", "epoch"), expected_keys)
    if any(not math.isfinite(row["val_loss"]) for row in final_rows):
        raise ValueError("Model selection requires finite validation losses")
    mean_val_loss = {
        configuration.code: sum(row["val_loss"] for row in final_rows if row["config"] == configuration.code) / len(SEEDS)
        for configuration in CONFIGURATIONS
    }
    selected_config = min(mean_val_loss, key=lambda code: (mean_val_loss[code], code))
    selected_row = min(
        (row for row in final_rows if row["config"] == selected_config),
        key=lambda row: (row["val_loss"], row["seed"]),
    )
    return {
        "epoch": 100,
        "mean_val_loss": mean_val_loss,
        "config": selected_config,
        "seed": selected_row["seed"],
        "val_loss": selected_row["val_loss"],
    }


def run_all_experiments(data: PreparedData) -> ExperimentSuite:
    metrics = []
    gradients = []
    models = {}
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            print(f"Training {configuration.code}, seed {seed}: 100 epochs", flush=True)
            experiment = run_experiment(configuration, seed, data)
            metrics.extend(experiment.metrics)
            gradients.extend(experiment.gradients)
            models[(configuration.code, seed)] = experiment.model
            last = experiment.metrics[-1]
            print(
                f"Completed {configuration.code}, seed {seed}: "
                f"validation loss={last['val_loss']:.6f}, accuracy={last['val_accuracy']:.6f}",
                flush=True,
            )
    validate_results(metrics, gradients)
    final_metrics = [row for row in metrics if row["epoch"] == 100]
    selection = select_run(final_metrics)
    selected_model = models[(selection["config"], selection["seed"])]
    test_metrics = evaluate(selected_model, data.test.features, data.test.labels)
    selection.update({"test_loss": test_metrics.loss, "test_accuracy": test_metrics.accuracy})
    return ExperimentSuite(metrics, gradients, final_metrics, selection, selected_model)
