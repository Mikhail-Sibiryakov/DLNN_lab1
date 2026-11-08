import argparse
import csv
import json
from pathlib import Path

import torch

from data import build_diagnostics, prepare_data
from experiments import GRADIENT_FIELDS, METRIC_FIELDS, run_all_experiments
from model import CONFIGURATIONS, check_model
from training import run_experiment
from visualization import generate_artifacts


def save_json(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output_path.write_text(serialized_report + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")


def save_csv(output_path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP laboratory: 12 experiments, validation selection and test")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--checks-only",
        action="store_true",
        help="Run only data and untrained model checks",
    )
    checks.add_argument(
        "--check-training",
        action="store_true",
        help="Run one training verification: D, seed 11, 100 epochs, without test evaluation",
    )
    checks.add_argument(
        "--plots-only",
        action="store_true",
        help="Create plots and tables from saved experiment results without training",
    )
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    data = prepare_data()
    if arguments.plots_only:
        generate_artifacts(data, Path("results"), Path("figures"))
        return
    diagnostics = build_diagnostics(data)
    failed_checks = [name for name, passed in diagnostics["checks"].items() if not passed]
    if failed_checks:
        raise ValueError("Data preparation checks failed: " + ", ".join(failed_checks))

    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False))
    save_json(Path("results/data_diagnostics.json"), diagnostics)

    model_checks = [
        check_model(configuration, seed, data.train.features[:8], data.train.labels[:8])
        for configuration in CONFIGURATIONS
        for seed in (11, 29)
    ]
    save_json(Path("results/model_checks.json"), {"models": model_checks})
    for result in model_checks:
        print(
            f"Model check {result['config']}, seed {result['seed']}: "
            f"parameters={result['parameter_count']}, "
            f"hidden layers={result['depth']}, all checks passed"
        )

    if arguments.check_training:
        configuration = next(configuration for configuration in CONFIGURATIONS if configuration.code == "D")
        experiment = run_experiment(configuration, 11, data)
        save_json(Path("results/training_check.json"), {
            "verification_only": True,
            "config": configuration.code,
            "seed": 11,
            "metrics": experiment.metrics,
            "gradients": experiment.gradients,
        })
        first, last = experiment.metrics[0], experiment.metrics[-1]
        print(
            f"Training check D, seed 11: train loss {first['train_loss']:.6f} -> {last['train_loss']:.6f}, "
            f"validation loss {first['val_loss']:.6f} -> {last['val_loss']:.6f}; "
            f"{len(experiment.metrics)} metric records, {len(experiment.gradients)} gradient records"
        )
    elif not arguments.checks_only:
        suite = run_all_experiments(data)
        save_csv(Path("results/metrics.csv"), suite.metrics, METRIC_FIELDS)
        save_csv(Path("results/gradients.csv"), suite.gradients, GRADIENT_FIELDS)
        save_csv(Path("results/final_metrics.csv"), suite.final_metrics, METRIC_FIELDS)
        save_json(Path("results/selection.json"), suite.selection)
        checkpoint_path = Path("results/selected_model.pt")
        torch.save({
            "config": suite.selection["config"],
            "seed": suite.selection["seed"],
            "epoch": suite.selection["epoch"],
            "state_dict": suite.selected_model.state_dict(),
        }, checkpoint_path)
        print(f"Saved: {checkpoint_path}")
        print(
            f"Selected {suite.selection['config']}, seed {suite.selection['seed']}, epoch 100: "
            f"test loss={suite.selection['test_loss']:.6f}, accuracy={suite.selection['test_accuracy']:.6f}"
        )
        generate_artifacts(data, Path("results"), Path("figures"))


if __name__ == "__main__":
    main()
