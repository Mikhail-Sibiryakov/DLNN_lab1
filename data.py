from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


@dataclass
class DataSplit:
    features: torch.Tensor
    labels: torch.Tensor


@dataclass
class PreparedData:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    probe: DataSplit
    original_train_features: np.ndarray
    scaler: StandardScaler


def make_split(features: np.ndarray, labels: np.ndarray, scaler: StandardScaler) -> DataSplit:
    return DataSplit(
        features=torch.as_tensor(scaler.transform(features), dtype=torch.float32, device="cpu"),
        labels=torch.as_tensor(labels, dtype=torch.long, device="cpu"),
    )


def prepare_data() -> PreparedData:
    features, labels = make_moons(n_samples=2400, noise=0.25, random_state=2026)
    remaining_features, test_features, remaining_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=2026, stratify=labels
    )
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        remaining_features,
        remaining_labels,
        test_size=0.25,
        random_state=2026,
        stratify=remaining_labels,
    )
    scaler = StandardScaler().fit(train_features)
    train = make_split(train_features, train_labels, scaler)
    return PreparedData(
        train=train,
        validation=make_split(validation_features, validation_labels, scaler),
        test=make_split(test_features, test_labels, scaler),
        probe=DataSplit(train.features[:128], train.labels[:128]),
        original_train_features=train_features,
        scaler=scaler,
    )


def build_diagnostics(data: PreparedData) -> dict:
    splits = {"train": data.train, "validation": data.validation, "test": data.test}
    expected_sizes = {"train": 1440, "validation": 480, "test": 480}
    split_statistics = {}
    for name, split in splits.items():
        classes, counts = torch.unique(split.labels, return_counts=True)
        split_statistics[name] = {
            "size": len(split.labels),
            "class_counts": dict(zip(map(str, classes.tolist()), counts.tolist(), strict=True)),
        }

    normalized_train = data.train.features.numpy()
    train_mean = normalized_train.mean(axis=0, dtype=np.float64)
    train_std = normalized_train.std(axis=0, dtype=np.float64)
    scaler_mean = data.scaler.mean_
    checks = {
        "split_sizes": all(len(split.labels) == expected_sizes[name] for name, split in splits.items()),
        "feature_shapes": all(
            split.features.shape == (expected_sizes[name], 2) for name, split in splits.items()
        ),
        "both_classes": all(torch.unique(split.labels).tolist() == [0, 1] for split in splits.values()),
        "feature_dtype_float32": all(split.features.dtype == torch.float32 for split in splits.values()),
        "label_dtype_long": all(split.labels.dtype == torch.long for split in splits.values()),
        "device_cpu": all(
            split.features.device.type == split.labels.device.type == "cpu" for split in splits.values()
        ),
        "features_finite": all(torch.isfinite(split.features).all().item() for split in splits.values()),
        "scaler_fit_on_train": data.scaler.n_samples_seen_ == 1440
        and scaler_mean is not None
        and np.allclose(scaler_mean, data.original_train_features.mean(axis=0), atol=1e-12),
        "train_mean_near_zero": np.allclose(train_mean, 0, atol=1e-6),
        "train_std_near_one": np.allclose(train_std, 1, atol=1e-6),
        "probe_preserves_train_prefix": torch.equal(data.probe.features, data.train.features[:128])
        and torch.equal(data.probe.labels, data.train.labels[:128]),
    }
    return {
        "dataset": {"n_samples": 2400, "noise": 0.25, "random_state": 2026},
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "splits": split_statistics,
        "scaler_fit_samples": int(data.scaler.n_samples_seen_),
        "train_feature_mean": train_mean.tolist(),
        "train_feature_std": train_std.tolist(),
        "probe_size": len(data.probe.labels),
        "checks": {name: bool(passed) for name, passed in checks.items()},
    }
