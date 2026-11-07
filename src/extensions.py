"""
Extensions for Robust ML-Based Map Matching Under GNSS Degradation.

Extension 1:
    Test independent Gaussian noise, persistent GPS bias,
    and temporally correlated GPS drift.

Extension 2:
    Evaluate on unseen trajectories using a trajectory-level split.

Important:
    Ground-truth labels must come from your existing map-matching
    pipeline. Nearest-road labels are proxy labels, not verified
    vehicle road assignments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXTENSION 1: GNSS DEGRADATION MODELS
# ============================================================

def add_gaussian_noise(
    xy,
    sigma_m,
    rng,
):
    """
    Independent Gaussian positioning noise.

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        Projected coordinates in metres.
    sigma_m : float
        Standard deviation of each coordinate's noise.
    rng : numpy random generator
    """
    noise = rng.normal(
        loc=0.0,
        scale=sigma_m,
        size=xy.shape,
    )

    return xy + noise


def add_persistent_bias(
    xy,
    bias_m,
    rng,
):
    """
    Apply a constant positioning offset throughout a trajectory.

    The offset direction is randomly selected, but its magnitude
    remains approximately constant throughout the trip.
    """
    angle = rng.uniform(0, 2 * np.pi)

    offset = np.array([
        bias_m * np.cos(angle),
        bias_m * np.sin(angle),
    ])

    return xy + offset


def add_correlated_drift(
    xy,
    sigma_m,
    rng,
    correlation=0.95,
):
    """
    Generate temporally correlated GNSS errors using an AR(1)
    process.

    Consecutive errors are correlated, simulating gradual
    positioning drift rather than independent point noise.

    sigma_m controls the approximate stationary standard
    deviation of each coordinate's error.
    """
    n = len(xy)

    if n == 0:
        return xy.copy()

    if not 0 <= correlation < 1:
        raise ValueError(
            "correlation must be >= 0 and < 1"
        )

    errors = np.zeros((n, 2), dtype=float)

    errors[0] = rng.normal(
        0,
        sigma_m,
        size=2,
    )

    innovation_scale = sigma_m * np.sqrt(
        1 - correlation**2
    )

    for i in range(1, n):
        innovation = rng.normal(
            0,
            innovation_scale,
            size=2,
        )

        errors[i] = (
            correlation * errors[i - 1]
            + innovation
        )

    return xy + errors


def generate_degraded_trajectory(
    xy,
    degradation_type,
    severity_m,
    seed=42,
):
    """
    Generate one reproducible degraded trajectory.

    degradation_type:
        gaussian
        persistent_bias
        correlated_drift
    """
    xy = np.asarray(xy, dtype=float)

    rng = np.random.default_rng(seed)

    if degradation_type == "gaussian":
        return add_gaussian_noise(
            xy,
            severity_m,
            rng,
        )

    if degradation_type == "persistent_bias":
        return add_persistent_bias(
            xy,
            severity_m,
            rng,
        )

    if degradation_type == "correlated_drift":
        return add_correlated_drift(
            xy,
            severity_m,
            rng,
        )

    raise ValueError(
        f"Unknown degradation type: {degradation_type}"
    )


def run_degradation_experiments(
    xy,
    predict_function,
    true_labels,
    repetitions=5,
):
    """
    Evaluate the existing map-matching predictor under
    different GNSS degradation types.

    Parameters
    ----------
    xy:
        Original projected coordinates, in metres.

    predict_function:
        Callable accepting degraded coordinates and returning
        predicted labels.

        Example:
            predict_function(noisy_xy)

        Connect this function to your existing map-matching
        implementation.

    true_labels:
        Reference labels for the original trajectory.

    repetitions:
        Number of independent random trials per setting.

    Returns
    -------
    DataFrame with accuracy and macro-F1 for each condition.
    """
    xy = np.asarray(xy, dtype=float)
    true_labels = np.asarray(true_labels)

    if len(xy) != len(true_labels):
        raise ValueError(
            "Coordinates and labels must have equal lengths."
        )

    degradation_types = [
        "gaussian",
        "persistent_bias",
        "correlated_drift",
    ]

    severities = [5, 10, 20]

    rows = []

    for degradation in degradation_types:
        for severity in severities:
            for repetition in range(repetitions):

                noisy_xy = generate_degraded_trajectory(
                    xy=xy,
                    degradation_type=degradation,
                    severity_m=severity,
                    seed=1000 + repetition,
                )

                predicted = np.asarray(
                    predict_function(noisy_xy)
                )

                if len(predicted) != len(true_labels):
                    raise ValueError(
                        "predict_function returned an incorrect "
                        "number of predictions."
                    )

                rows.append({
                    "degradation": degradation,
                    "severity_m": severity,
                    "repetition": repetition + 1,
                    "accuracy": accuracy_score(
                        true_labels,
                        predicted,
                    ),
                    "macro_f1": f1_score(
                        true_labels,
                        predicted,
                        average="macro",
                        zero_division=0,
                    ),
                })

    results = pd.DataFrame(rows)

    results.to_csv(
        TABLES / "degradation_trials.csv",
        index=False,
    )

    summary = (
        results
        .groupby(["degradation", "severity_m"])
        [["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(
        TABLES / "degradation_summary.csv",
        index=False,
    )

    plot_degradation_results(results)

    return results, summary


def plot_degradation_results(results):
    """
    Save an accuracy comparison plot.
    """
    summary = (
        results
        .groupby(["degradation", "severity_m"])["accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for degradation, group in summary.groupby("degradation"):
        group = group.sort_values("severity_m")

        plt.plot(
            group["severity_m"],
            group["accuracy"] * 100,
            marker="o",
            label=degradation.replace("_", " ").title(),
        )

    plt.xlabel("Error severity (metres)")
    plt.ylabel("Accuracy (%)")
    plt.title("Map-Matching Accuracy Under GNSS Degradation")
    plt.xticks([5, 10, 20])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES / "degradation_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# EXTENSION 2: UNSEEN-TRAJECTORY GENERALIZATION
# ============================================================

def evaluate_unseen_trajectories(
    X,
    y,
    trajectory_ids,
    random_state=42,
):
    """
    Train a Random Forest on complete trajectories and evaluate
    on entirely held-out trajectories.

    Parameters
    ----------
    X:
        Feature matrix produced by the existing pipeline.

    y:
        Reference labels produced by the existing pipeline.

    trajectory_ids:
        One trajectory identifier per feature row.

        All points belonging to the same trajectory must share
        the same identifier.

    Returns
    -------
    model, metrics, predictions

    IMPORTANT:
        Do not pass one trajectory with different IDs assigned
        to individual points. IDs must represent actual trips.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    trajectory_ids = np.asarray(trajectory_ids)

    if not (
        len(X) == len(y) == len(trajectory_ids)
    ):
        raise ValueError(
            "X, y, and trajectory_ids must have equal lengths."
        )

    unique_trajectories = np.unique(trajectory_ids)

    if len(unique_trajectories) < 2:
        raise ValueError(
            "At least two distinct trajectories are required. "
            "Load multiple real trips before running this test."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=trajectory_ids,
        )
    )

    train_groups = set(trajectory_ids[train_idx])
    test_groups = set(trajectory_ids[test_idx])

    assert train_groups.isdisjoint(test_groups), (
        "Data leakage: a trajectory appears in both splits."
    )

    # Scale numerical features and train the classifier.
    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(
        X[train_idx],
        y[train_idx],
    )

    predictions = model.predict(X[test_idx])

    metrics = {
        "train_trajectory_count": len(train_groups),
        "test_trajectory_count": len(test_groups),
        "train_point_count": len(train_idx),
        "test_point_count": len(test_idx),
        "accuracy": accuracy_score(
            y[test_idx],
            predictions,
        ),
        "macro_f1": f1_score(
            y[test_idx],
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    pd.DataFrame([metrics]).to_csv(
        TABLES / "unseen_trajectory_metrics.csv",
        index=False,
    )

    report = classification_report(
        y[test_idx],
        predictions,
        zero_division=0,
    )

    with open(
        TABLES / "unseen_trajectory_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Evaluation on held-out trajectories\n\n"
        )

        f.write(
            f"Training trajectories: {len(train_groups)}\n"
        )

        f.write(
            f"Testing trajectories: {len(test_groups)}\n\n"
        )

        f.write(report)

    return model, metrics, {
        "test_indices": test_idx,
        "predictions": predictions,
        "true_labels": y[test_idx],
    }


# ============================================================
# OPTIONAL: PRINT EXPERIMENT RESULTS
# ============================================================

def print_metrics(metrics):
    """
    Print metrics in a readable format.
    """
    print("\nEXPERIMENT RESULTS")
    print("-" * 45)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")