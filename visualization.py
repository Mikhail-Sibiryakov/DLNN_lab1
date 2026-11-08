import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from data import PreparedData
from experiments import select_run, validate_results
from model import CONFIGURATIONS


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINE_STYLES = ("-", "--", "-.", ":", "-", "--")


@dataclass
class SavedResults:
    metrics: list[dict]
    gradients: list[dict]
    final_metrics: list[dict]
    selection: dict


def read_records(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {
                field: value if field == "config" else int(value) if field in {"seed", "epoch", "layer"} else float(value)
                for field, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def load_results(directory: Path) -> SavedResults:
    metrics = read_records(directory / "metrics.csv")
    gradients = read_records(directory / "gradients.csv")
    final_metrics = read_records(directory / "final_metrics.csv")
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    validate_results(metrics, gradients)
    expected_final = sorted(
        (row for row in metrics if row["epoch"] == 100), key=lambda row: (row["config"], row["seed"]),
    )
    if sorted(final_metrics, key=lambda row: (row["config"], row["seed"])) != expected_final:
        raise ValueError("Final metrics do not match epoch 100 in metrics.csv")
    expected_selection = select_run(final_metrics)
    if any(selection.get(field) != value for field, value in expected_selection.items()):
        raise ValueError("Saved selection does not match validation losses on epoch 100")
    if any(field not in selection or not math.isfinite(selection[field]) for field in ("test_loss", "test_accuracy")):
        raise ValueError("Saved test metrics must be finite")
    if not 0 <= selection["test_accuracy"] <= 1:
        raise ValueError("Saved test accuracy must be a fraction between 0 and 1")
    return SavedResults(metrics, gradients, final_metrics, selection)


def make_figure(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.2)
    return figure, axis


def plot_train_scatter(data: PreparedData) -> Figure:
    figure, axis = make_figure("Train до масштабирования", "$x_1$", "$x_2$")
    labels = data.train.labels.numpy()
    for label, color in enumerate(COLORS[:2]):
        features = data.original_train_features[labels == label]
        axis.scatter(
            features[:, 0], features[:, 1], s=14, alpha=0.65, edgecolors="none",
            color=color, label=f"Класс {label} ({len(features)})",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    return figure


def plot_validation(metrics: list[dict], metric: str) -> Figure:
    labels = {"val_loss": "Cross-entropy loss", "val_accuracy": "Accuracy"}
    figure, axis = make_figure(f"Validation {metric.removeprefix('val_')}, seed 11", "Эпоха", labels[metric])
    for configuration, color, style in zip(CONFIGURATIONS, COLORS, LINE_STYLES, strict=True):
        rows = sorted(
            (row for row in metrics if row["config"] == configuration.code and row["seed"] == 11),
            key=lambda row: row["epoch"],
        )
        axis.plot(
            [row["epoch"] for row in rows], [row[metric] for row in rows],
            label=configuration.code, color=color, linestyle=style, linewidth=1.5,
        )
    axis.set_xlim(0, 100)
    if metric == "val_accuracy":
        axis.set_ylim(0, 1)
    axis.legend(title="Конфигурация", ncols=3)
    return figure


def plot_gradients(gradients: list[dict], configuration: str) -> Figure:
    figure, axis = make_figure(
        f"Градиенты скрытых весов: {configuration}, seed 11",
        "Номер скрытого слоя", r"$\|\partial\mathcal{L}_{\mathrm{probe}}/\partial W_l\|_F$",
    )
    selected_rows = [row for row in gradients if row["config"] == configuration and row["seed"] == 11]
    for epoch, color in zip((0, 20, 100), COLORS[:3], strict=True):
        rows = sorted((row for row in selected_rows if row["epoch"] == epoch), key=lambda row: row["layer"])
        norms = np.array([row["gradient_norm"] for row in rows])
        axis.plot(
            [row["layer"] for row in rows], np.where(norms > 0, norms, np.nan),
            marker="o", markersize=4, color=color, linewidth=1.6, label=f"Эпоха {epoch}",
        )
    axis.set_yscale("log")
    positive_norms = [row["gradient_norm"] for row in gradients if row["seed"] == 11 and row["gradient_norm"] > 0]
    if positive_norms:
        lower = 10 ** math.floor(math.log10(min(positive_norms)))
        upper = 10 ** math.ceil(math.log10(max(positive_norms)))
        if lower == upper:
            lower /= 10
            upper *= 10
        axis.set_ylim(lower, upper)
    else:
        axis.set_ylim(0.1, 1)
    if all(row["gradient_norm"] == 0 for row in selected_rows):
        axis.text(0.5, 0.5, "Все нормы равны нулю; см. таблицу", transform=axis.transAxes, ha="center")
    elif any(row["gradient_norm"] == 0 for row in selected_rows):
        axis.text(0.02, 0.02, "Нулевые нормы пропущены; см. таблицу", transform=axis.transAxes, fontsize=9)
    axis.set_xticks(range(1, 7))
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    return figure


def plot_selected_loss(metrics: list[dict], selection: dict) -> Figure:
    rows = sorted(
        (row for row in metrics if row["config"] == selection["config"] and row["seed"] == selection["seed"]),
        key=lambda row: row["epoch"],
    )
    figure, axis = make_figure(
        f"Выбранный запуск: {selection['config']}, seed {selection['seed']}",
        "Эпоха", "Cross-entropy loss",
    )
    for metric, label, color in zip(("train_loss", "val_loss"), ("Train", "Validation"), COLORS[:2], strict=True):
        axis.plot([row["epoch"] for row in rows], [row[metric] for row in rows], label=label, color=color, linewidth=1.6)
    axis.set_xlim(0, 100)
    axis.legend()
    return figure


def write_tables(saved: SavedResults, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    means_path = directory / "mean_val_loss.csv"
    with means_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("config", "mean_val_loss"))
        writer.writeheader()
        writer.writerows(
            {"config": configuration.code, "mean_val_loss": saved.selection["mean_val_loss"][configuration.code]}
            for configuration in CONFIGURATIONS
        )
    lines = [
        "# Таблицы результатов", "",
        "Значения округлены до шести знаков только для отображения. Исходные значения сохранены в CSV/JSON.", "",
        "## Метрики на эпохе 100", "",
        "| config | seed | train_loss | train_accuracy | val_loss | val_accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(saved.final_metrics, key=lambda row: (row["config"], row["seed"])):
        values = " | ".join(f"{row[field]:.6f}" for field in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"))
        lines.append(f"| {row['config']} | {row['seed']} | {values} |")
    lines.extend(["", "## Средний validation loss на эпохе 100", "", "| config | mean_val_loss |", "| --- | ---: |"])
    for configuration in CONFIGURATIONS:
        lines.append(f"| {configuration.code} | {saved.selection['mean_val_loss'][configuration.code]:.6f} |")
    selection = saved.selection
    selected = next(row for row in saved.final_metrics if row["config"] == selection["config"] and row["seed"] == selection["seed"])
    lines.extend([
        "", "## Выбранный запуск", "",
        f"Конфигурация {selection['config']}, seed {selection['seed']}, эпоха {selection['epoch']}.", "",
        "| Выборка | Loss | Accuracy |", "| --- | ---: | ---: |",
        f"| Train | {selected['train_loss']:.6f} | {selected['train_accuracy']:.6f} |",
        f"| Validation | {selected['val_loss']:.6f} | {selected['val_accuracy']:.6f} |",
        f"| Test | {selection['test_loss']:.6f} | {selection['test_accuracy']:.6f} |",
        "", "## Нулевые нормы градиентов", "",
    ])
    zeros = [row for row in saved.gradients if row["gradient_norm"] == 0]
    if zeros:
        lines.extend(["| config | seed | epoch | layer | gradient_norm |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in sorted(zeros, key=lambda row: (row["config"], row["seed"], row["epoch"], row["layer"])):
            lines.append(f"| {row['config']} | {row['seed']} | {row['epoch']} | {row['layer']} | 0.0 |")
    else:
        lines.append(f"Нулевых норм среди {len(saved.gradients)} измерений нет.")
    summary_path = directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [means_path, summary_path]


def save_figure(figure: Figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=200, facecolor="white")
    finally:
        plt.close(figure)
    return path


def generate_artifacts(data: PreparedData, results_directory: Path, figures_directory: Path) -> list[Path]:
    saved = load_results(results_directory)
    figures_directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "legend.fontsize": 10}):
        paths = [save_figure(plot_train_scatter(data), figures_directory / "train_scatter.png")]
        for metric, name in [("val_loss", "loss"), ("val_accuracy", "accuracy")]:
            paths.append(save_figure(plot_validation(saved.metrics, metric), figures_directory / f"validation_{name}_seed11.png"))
        for configuration in ("A", "C", "D"):
            path = figures_directory / f"gradients_{configuration}_seed11.png"
            paths.append(save_figure(plot_gradients(saved.gradients, configuration), path))
        paths.append(save_figure(plot_selected_loss(saved.metrics, saved.selection), figures_directory / "selected_loss.png"))
    paths.extend(write_tables(saved, results_directory))
    for path in paths:
        print(f"Saved: {path}")
    return paths
