"""Section 4.3 conditional and out-of-sample AMQR simulations.

This script evaluates the current population-aligned estimator: at every
predictor query it first fits an anchor-free transport from localized source
weights to an estimated intrinsic-volume measure, then selects the localized
Fréchet medoid and computes Wasserstein distances between conditional
transport rows.  No anchor is imposed on the transport problem.

Two designs are included.  The first is a smooth sphere with a conditional
density mode that moves with the predictor, which is aligned with the
smooth-manifold theory.  The second is a three-arm functional
trajectory tree whose conditional branch probabilities and within-branch
progress change with the predictor.  The tree is an application-motivated
stress test and is not covered by the smooth-manifold theory.

Both designs use a prespecified scientific reference state: a fixed spherical
prototype and the common trajectory origin.  The estimator receives predictors,
ambient pairwise response distances, and distances from this prototype.
Analytic sphere/tree distances and deterministic high-resolution support grids
are used only to construct the oracle ranks and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import beta as beta_distribution
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.anchored_uniformization import (
    conditional_transport_wasserstein_scores,
    estimate_intrinsic_dimension,
    estimate_uniform_volume_weights,
    extend_anchored_ranks,
    graph_geodesic_distances,
    intrinsic_uniformization_plan,
    pooled_conditional_transport_wasserstein_scores,
)
from _paper_simulation_utils import adaptive_neighbors, sphere_geodesic


DESIGNS = ("smooth_sphere", "branching_tree")
DESIGN_LABELS = {
    "smooth_sphere": "Smooth sphere",
    "branching_tree": "Branching functional trajectory",
}
METHODS = ("amqr", "intrinsic_radial", "ambient_radial", "global_amqr", "oracle")
PRIMARY_METHODS = ("amqr", "intrinsic_radial", "ambient_radial", "global_amqr")
METHOD_LABELS = {
    "amqr": "Conditional AMQR",
    "intrinsic_radial": "Conditional intrinsic radial",
    "ambient_radial": "Conditional ambient radial",
    "global_amqr": "Unconditional AMQR",
    "oracle": "Oracle",
}
METHOD_COLORS = {
    "amqr": "#D55E00",
    "intrinsic_radial": "#0072B2",
    "ambient_radial": "#56B4E9",
    "global_amqr": "#009E73",
    "oracle": "#777777",
}
METHOD_MARKERS = {
    "amqr": "o",
    "intrinsic_radial": "s",
    "ambient_radial": "^",
    "global_amqr": "D",
    "oracle": "x",
}
LEVELS = (0.25, 0.50, 0.80, 0.90)
BRANCH_ANGLES = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
BRANCH_CENTERS = np.asarray([0.10, 0.50, 0.90])
SPHERE_REFERENCE = np.asarray([1.0, 0.0, 0.0])


def parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in text.split(",") if value.strip()}))
    if not values or min(values) < 80:
        raise argparse.ArgumentTypeError("sample sizes must be at least 80")
    return values


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("query values must lie in (0,1)")
    return values


def weighted_cdf_at(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(reference_values, dtype=float).reshape(-1)
    weights = np.asarray(reference_weights, dtype=float).reshape(-1)
    weights = weights / np.sum(weights)
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    positions = np.searchsorted(values[order], query_values, side="right") - 1
    output = np.zeros_like(np.asarray(query_values, dtype=float))
    valid = positions >= 0
    output[valid] = cumulative[positions[valid]]
    return np.clip(output, 0.0, 1.0)


def weighted_cdf_levels(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weighted_cdf_at(values, weights, values)


def interpolate_raw_scores(
    cross_distances: np.ndarray,
    train_scores: np.ndarray,
    train_weights: np.ndarray,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cross = np.asarray(cross_distances, dtype=float)
    k = min(max(3, int(k_neighbors)), cross.shape[1])
    neighbours = np.argpartition(cross, kth=k - 1, axis=1)[:, :k]
    distances = np.take_along_axis(cross, neighbours, axis=1)
    bandwidth = np.maximum(np.max(distances, axis=1), 1e-12)
    local_weights = np.exp(-0.5 * (distances / bandwidth[:, None]) ** 2)
    local_weights /= np.sum(local_weights, axis=1, keepdims=True)
    raw = np.sum(local_weights * np.asarray(train_scores)[neighbours], axis=1)
    ranks = weighted_cdf_at(train_scores, train_weights, raw)
    return ranks, raw


def kernel_weights(x_train: np.ndarray, query_x: float, bandwidth: float) -> np.ndarray:
    standardized = (np.asarray(x_train, dtype=float) - float(query_x)) / float(bandwidth)
    weights = np.exp(-0.5 * standardized ** 2)
    weights = np.maximum(weights, 1e-300)
    return weights / np.sum(weights)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def region_mask(ranks: np.ndarray, level: float) -> np.ndarray:
    return np.asarray(ranks, dtype=float) <= float(level)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    return float(np.sum(left & right) / np.sum(union)) if np.any(union) else 1.0


def sphere_direction(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    longitude = 2.0 * np.pi * (0.05 + 0.90 * x)
    latitude = 0.45 * np.sin(2.0 * np.pi * x)
    return np.column_stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def sample_sphere(x: np.ndarray, rng: np.random.Generator, density_ratio: float) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    beta = 0.5 * math.log(float(density_ratio))
    points = np.empty((len(x), 3), dtype=float)
    directions = sphere_direction(x)
    unresolved = np.arange(len(x))
    while unresolved.size:
        proposals = rng.normal(size=(len(unresolved), 3))
        proposals /= np.linalg.norm(proposals, axis=1, keepdims=True)
        score = np.sum(proposals * directions[unresolved], axis=1)
        accepted = rng.random(len(unresolved)) <= np.exp(beta * (score - 1.0))
        points[unresolved[accepted]] = proposals[accepted]
        unresolved = unresolved[~accepted]
    density = np.exp(beta * np.sum(points * directions, axis=1))
    return {
        "x": x,
        "points": points,
        "theta": np.full(len(x), np.nan),
        "density": density,
        "branch": np.full(len(x), -1, dtype=int),
        "amplitude": np.full(len(x), np.nan),
    }


def branch_probabilities(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    scores = np.exp(-0.5 * ((x[:, None] - BRANCH_CENTERS[None, :]) / 0.22) ** 2)
    scores /= np.sum(scores, axis=1, keepdims=True)
    return 0.08 + 0.76 * scores


def amplitude_means(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack(
        [
            0.25 + 0.55 * (1.0 - x),
            0.30 + 0.45 * np.maximum(1.0 - 2.0 * np.abs(x - 0.5), 0.0),
            0.25 + 0.55 * x,
        ]
    )


def trajectory_basis(grid: np.ndarray) -> np.ndarray:
    onset = 0.30
    scaled = np.clip((grid - onset) / (1.0 - onset), 0.0, 1.0)
    first = scaled ** 2 * (3.0 - 2.0 * scaled)
    first /= np.sqrt(np.mean(first ** 2))
    second = first * np.sin(2.0 * np.pi * grid)
    second -= np.mean(second * first) * first
    second /= np.sqrt(np.mean(second ** 2))
    return np.vstack([first, second])


def branch_density(
    x: np.ndarray,
    branch: np.ndarray,
    amplitude: np.ndarray,
    concentration: float,
    uniform_fraction: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    branch = np.asarray(branch, dtype=int).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=float).reshape(-1)
    mass = branch_probabilities(x)[np.arange(len(x)), branch]
    means = amplitude_means(x)[np.arange(len(x)), branch]
    alpha = means * float(concentration)
    beta_parameter = (1.0 - means) * float(concentration)
    mixture = float(uniform_fraction) + (1.0 - float(uniform_fraction)) * beta_distribution.pdf(
        np.clip(amplitude, 1e-10, 1.0 - 1e-10), alpha, beta_parameter
    )
    return mass * mixture


def sample_tree(
    x: np.ndarray,
    rng: np.random.Generator,
    grid: np.ndarray,
    concentration: float,
    uniform_fraction: float,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    probabilities = branch_probabilities(x)
    branch = np.sum(rng.random(len(x))[:, None] > np.cumsum(probabilities, axis=1), axis=1)
    branch = np.minimum(branch, 2).astype(int)
    means = amplitude_means(x)[np.arange(len(x)), branch]
    alpha = means * float(concentration)
    beta_parameter = (1.0 - means) * float(concentration)
    amplitude = rng.beta(alpha, beta_parameter)
    use_uniform = rng.random(len(x)) < float(uniform_fraction)
    amplitude[use_uniform] = rng.random(np.sum(use_uniform))
    directions = np.column_stack(
        [np.cos(BRANCH_ANGLES[branch]), np.sin(BRANCH_ANGLES[branch])]
    )
    points = amplitude[:, None] * (directions @ trajectory_basis(grid))
    density = branch_density(
        x, branch, amplitude, concentration, uniform_fraction
    )
    return {
        "x": x,
        "points": points,
        "theta": np.full(len(x), np.nan),
        "density": density,
        "branch": branch,
        "amplitude": amplitude,
    }


def true_distances(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], design: str) -> np.ndarray:
    if design == "smooth_sphere":
        return sphere_geodesic(np.asarray(left["points"]), np.asarray(right["points"]))
    left_branch = np.asarray(left["branch"], dtype=int)
    right_branch = np.asarray(right["branch"], dtype=int)
    left_amplitude = np.asarray(left["amplitude"], dtype=float)
    right_amplitude = np.asarray(right["amplitude"], dtype=float)
    same = left_branch[:, None] == right_branch[None, :]
    return np.where(
        same,
        np.abs(left_amplitude[:, None] - right_amplitude[None, :]),
        left_amplitude[:, None] + right_amplitude[None, :],
    )


def oracle_support(
    design: str,
    query_x: float,
    sphere_grid_size: int,
    tree_grid_per_branch: int,
    sphere_density_ratio: float,
    concentration: float,
    uniform_fraction: float,
) -> Dict[str, Any]:
    if design == "smooth_sphere":
        index = np.arange(sphere_grid_size, dtype=float)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (index + 0.5) / sphere_grid_size
        radius = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
        points = np.column_stack(
            [radius * np.cos(golden_angle * index), radius * np.sin(golden_angle * index), z]
        )
        support = {
            "points": points,
            "theta": np.full(sphere_grid_size, np.nan),
            "branch": np.full(sphere_grid_size, -1, dtype=int),
            "amplitude": np.full(sphere_grid_size, np.nan),
        }
        beta = 0.5 * math.log(float(sphere_density_ratio))
        source = np.exp(beta * (points @ sphere_direction(np.asarray([query_x]))[0]))
    else:
        amplitude_grid = (np.arange(tree_grid_per_branch) + 0.5) / tree_grid_per_branch
        branch = np.repeat(np.arange(3), tree_grid_per_branch)
        amplitude = np.tile(amplitude_grid, 3)
        support = {
            "points": np.empty((len(branch), 0)),
            "theta": np.full(len(branch), np.nan),
            "branch": branch,
            "amplitude": amplitude,
        }
        source = branch_density(
            np.full(len(branch), query_x),
            branch,
            amplitude,
            concentration,
            uniform_fraction,
        )
    source = np.asarray(source, dtype=float)
    source /= np.sum(source)
    target = np.full(len(source), 1.0 / len(source))
    distances = true_distances(support, support, design)
    if design == "smooth_sphere":
        anchor = int(np.argmin(np.linalg.norm(support["points"] - SPHERE_REFERENCE[None, :], axis=1)))
    else:
        anchor = int(np.argmin(support["amplitude"]))
    plan, plan_diagnostics = intrinsic_uniformization_plan(distances, source, target)
    scores, score_diagnostics = conditional_transport_wasserstein_scores(
        distances, plan, source, anchor
    )
    return {
        "design": design,
        "query_x": float(query_x),
        "support": support,
        "distances": distances,
        "source": source,
        "target": target,
        "center": int(anchor),
        "scores": scores,
        "ranks": weighted_cdf_levels(scores, source),
        "plan_diagnostics": plan_diagnostics,
        "score_diagnostics": score_diagnostics,
    }


def oracle_test_ranks(
    oracle: Mapping[str, Any],
    test: Mapping[str, np.ndarray],
    interpolation_neighbors: int,
) -> np.ndarray:
    cross = true_distances(test, oracle["support"], str(oracle["design"]))
    ranks, _ = interpolate_raw_scores(
        cross, oracle["scores"], oracle["source"], interpolation_neighbors
    )
    return ranks


def fit_geometry(points: np.ndarray, true_dimension: int) -> Dict[str, Any]:
    local = cdist(points, points)
    dim_k, graph_k, volume_k = adaptive_neighbors(len(points), true_dimension)
    dimension = estimate_intrinsic_dimension(local, k_neighbors=dim_k)
    intrinsic, graph_diagnostics = graph_geodesic_distances(
        local,
        k_neighbors=graph_k,
        ensure_connected=True,
        maximum_neighbors=max(50, graph_k),
    )
    volume, volume_diagnostics = estimate_uniform_volume_weights(
        local,
        intrinsic_dimension=dimension,
        k_neighbors=volume_k,
        winsor_quantile=0.0,
    )
    return {
        "local": local,
        "intrinsic": intrinsic,
        "dimension": dimension,
        "graph_k": graph_k,
        "volume_k": volume_k,
        "dimension_k": dim_k,
        "graph_diagnostics": graph_diagnostics,
        "volume": volume,
        "volume_diagnostics": volume_diagnostics,
    }


def fit_scores(
    intrinsic: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    anchor: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    plan, plan_diagnostics = intrinsic_uniformization_plan(
        intrinsic, source, target, entropic_regularization=0.0
    )
    scores, score_diagnostics = pooled_conditional_transport_wasserstein_scores(
        intrinsic, plan, source, anchor
    )
    return weighted_cdf_levels(scores, source), scores, {
        "runtime_seconds": float(time.perf_counter() - started),
        "plan": plan_diagnostics,
        "score": score_diagnostics,
    }


def scientific_anchor_index(train: Mapping[str, np.ndarray], design: str) -> int:
    """Represent the prespecified reference state by its nearest observation."""
    if design == "smooth_sphere":
        return int(
            np.argmin(
                np.linalg.norm(
                    np.asarray(train["points"]) - SPHERE_REFERENCE[None, :], axis=1
                )
            )
        )
    # The tree reference is the observed zero function.  Selection therefore
    # uses its directly computable ambient distance, not the latent amplitude.
    return int(np.argmin(np.linalg.norm(np.asarray(train["points"]), axis=1)))


def center_error(
    train: Mapping[str, np.ndarray],
    estimated_index: int,
    oracle: Mapping[str, Any],
) -> float:
    design = str(oracle["design"])
    one = {key: np.asarray(train[key])[[estimated_index]] for key in ("points", "theta", "branch", "amplitude")}
    support = oracle["support"]
    oracle_center = int(oracle["center"])
    other = {key: np.asarray(support[key])[[oracle_center]] for key in ("points", "theta", "branch", "amplitude")}
    return float(true_distances(one, other, design)[0, 0])


def evaluate(
    method: str,
    ranks: np.ndarray,
    oracle_ranks: np.ndarray,
    design: str,
    query_x: float,
    test: Mapping[str, np.ndarray],
    estimated_center_error: float,
    local_ess: float,
    runtime_seconds: float,
) -> Dict[str, Any]:
    ranks = np.asarray(ranks, dtype=float)
    oracle_ranks = np.asarray(oracle_ranks, dtype=float)
    record: Dict[str, Any] = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "rank_spearman": finite_spearman(ranks, oracle_ranks),
        "rank_mae": float(np.mean(np.abs(ranks - oracle_ranks))),
        "center_error": float(estimated_center_error),
        "local_effective_sample_size": float(local_ess),
        "runtime_seconds": float(runtime_seconds),
    }
    coverage = []
    overlaps = []
    for level in LEVELS:
        predicted = region_mask(ranks, level)
        truth = region_mask(oracle_ranks, level)
        record[f"coverage_error_{int(100 * level)}"] = abs(float(np.mean(predicted)) - level)
        record[f"jaccard_{int(100 * level)}"] = jaccard(predicted, truth)
        coverage.append(record[f"coverage_error_{int(100 * level)}"])
        overlaps.append(record[f"jaccard_{int(100 * level)}"])
    record["mean_coverage_error"] = float(np.mean(coverage))
    record["max_coverage_error"] = float(np.max(coverage))
    record["mean_region_jaccard"] = float(np.mean(overlaps))
    predicted_50 = region_mask(ranks, 0.50)
    oracle_50 = region_mask(oracle_ranks, 0.50)
    record["false_inclusion_50"] = float(np.mean(predicted_50 & ~oracle_50))
    if design == "branching_tree":
        rare_branch = int(np.argmin(branch_probabilities(np.asarray([query_x]))[0]))
        eligible = region_mask(oracle_ranks, 0.80) & (np.asarray(test["branch"]) == rare_branch)
        record["rare_branch_recall_80"] = (
            float(np.mean(region_mask(ranks, 0.80)[eligible])) if np.any(eligible) else math.nan
        )
    else:
        record["rare_branch_recall_80"] = math.nan
    return record


def run_query(
    design: str,
    query_x: float,
    train: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    oracle: Mapping[str, Any],
    geometry: Mapping[str, Any],
    bandwidth: float,
    interpolation_neighbors: int,
    graph_connections: int,
    global_fit: Mapping[str, Any],
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source = kernel_weights(train["x"], query_x, bandwidth)
    local_ess = float(1.0 / np.sum(source ** 2))
    effective_interpolation_neighbors = max(
        int(interpolation_neighbors), int(np.ceil(np.sqrt(local_ess)))
    )
    intrinsic = np.asarray(geometry["intrinsic"])
    center = scientific_anchor_index(train, design)
    amqr_train_ranks, amqr_scores, amqr_diagnostics = fit_scores(
        intrinsic, source, geometry["volume"], center
    )
    ambient_cross = cdist(test["points"], train["points"])
    amqr_oos, _, graph_cross, extension_diagnostics = extend_anchored_ranks(
        ambient_cross,
        intrinsic,
        amqr_scores,
        source,
        graph_distance_scale=float(geometry["graph_diagnostics"]["median_graph_distance_before_scaling"]),
        graph_connections=graph_connections,
        interpolation_neighbors=effective_interpolation_neighbors,
    )
    intrinsic_ranks = weighted_cdf_at(intrinsic[center], source, graph_cross[:, center])
    ambient_ranks = weighted_cdf_at(geometry["local"][center], source, ambient_cross[:, center])
    global_ranks, _ = interpolate_raw_scores(
        graph_cross,
        global_fit["scores"],
        global_fit["source"],
        interpolation_neighbors,
    )
    oracle_ranks = oracle_test_ranks(oracle, test, interpolation_neighbors)
    estimated_center_error = center_error(train, center, oracle)
    global_center_error = center_error(train, int(global_fit["center"]), oracle)
    method_values = {
        "amqr": (amqr_oos, estimated_center_error, amqr_diagnostics["runtime_seconds"] + extension_diagnostics["runtime_seconds"]),
        "intrinsic_radial": (intrinsic_ranks, estimated_center_error, extension_diagnostics["runtime_seconds"]),
        "ambient_radial": (ambient_ranks, estimated_center_error, 0.0),
        "global_amqr": (global_ranks, global_center_error, global_fit["runtime_seconds"]),
        "oracle": (oracle_ranks, 0.0, 0.0),
    }
    rows = [
        evaluate(
            method,
            values[0],
            oracle_ranks,
            design,
            query_x,
            test,
            float(values[1]),
            local_ess,
            float(values[2]),
        )
        for method, values in method_values.items()
    ]
    details = {
        "center": int(center),
        "center_error": estimated_center_error,
        "local_ess": local_ess,
        "amqr_diagnostics": amqr_diagnostics,
        "extension_diagnostics": extension_diagnostics,
        "example": {
            "design": design,
            "query_x": float(query_x),
            "test": test,
            "ranks": {method: values[0] for method, values in method_values.items()},
        },
    }
    return rows, details


def summarize_clustered(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_spearman", "rank_mae", "mean_coverage_error", "max_coverage_error",
        "mean_region_jaccard", "jaccard_25", "jaccard_50", "jaccard_80", "jaccard_90",
        "false_inclusion_50", "rare_branch_recall_80", "center_error",
        "local_effective_sample_size", "runtime_seconds",
    ]
    per_repeat = records.groupby(
        ["design", "design_label", "n_train", "method", "method_label", "repeat"],
        as_index=False,
    )[metrics].mean()
    rows: list[Dict[str, Any]] = []
    group_columns = ["design", "design_label", "n_train", "method", "method_label"]
    for keys, group in per_repeat.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["repeats"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            row[f"{metric}_se"] = (
                float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_x(records: pd.DataFrame) -> pd.DataFrame:
    maximum_n = int(records["n_train"].max())
    selected = records[records["n_train"] == maximum_n]
    metrics = ["rank_mae", "max_coverage_error", "jaccard_50", "false_inclusion_50", "rare_branch_recall_80"]
    rows: list[Dict[str, Any]] = []
    for keys, group in selected.groupby(["design", "query_x", "method"], sort=True):
        row = dict(zip(["design", "query_x", "method"], keys))
        row["n_train"] = maximum_n
        row["repeats"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            row[f"{metric}_se"] = (
                float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(records: pd.DataFrame) -> pd.DataFrame:
    metrics = ("rank_mae", "max_coverage_error", "jaccard_50", "false_inclusion_50", "rare_branch_recall_80")
    comparisons = (("amqr", "intrinsic_radial"), ("amqr", "ambient_radial"), ("amqr", "global_amqr"))
    per_repeat = records.groupby(["design", "n_train", "repeat", "method"], as_index=False)[list(metrics)].mean()
    rows: list[Dict[str, Any]] = []
    for design in DESIGNS:
        for n_train in sorted(records["n_train"].unique()):
            subset = per_repeat[(per_repeat["design"] == design) & (per_repeat["n_train"] == n_train)]
            for left_method, right_method in comparisons:
                left = subset[subset["method"] == left_method].set_index("repeat")
                right = subset[subset["method"] == right_method].set_index("repeat")
                common = left.index.intersection(right.index)
                for metric in metrics:
                    difference = left.loc[common, metric].to_numpy(float) - right.loc[common, metric].to_numpy(float)
                    difference = difference[np.isfinite(difference)]
                    mean = float(np.mean(difference)) if difference.size else math.nan
                    se = float(np.std(difference, ddof=1) / np.sqrt(difference.size)) if difference.size > 1 else 0.0
                    rows.append({
                        "design": design,
                        "n_train": int(n_train),
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "mean_difference": mean,
                        "standard_error": se,
                        "ci95_lower": mean - 1.96 * se,
                        "ci95_upper": mean + 1.96 * se,
                        "paired_repeats": int(difference.size),
                    })
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    axis.tick_params(labelsize=8)


def plot_recovery(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True)
    panels = (
        ("rank_mae_mean", "rank_mae_se", "OOS rank MAE", (0.0, None)),
        ("jaccard_50_mean", "jaccard_50_se", "50% region Jaccard", (0.0, 1.03)),
        ("max_coverage_error_mean", "max_coverage_error_se", "Maximum coverage error", (0.0, None)),
    )
    for row, design in enumerate(DESIGNS):
        for column, (metric, error, label, limits) in enumerate(panels):
            axis = axes[row, column]
            subset = summary[summary["design"] == design]
            for method in PRIMARY_METHODS:
                values = subset[subset["method"] == method].sort_values("n_train")
                axis.errorbar(
                    values["n_train"], values[metric], yerr=1.96 * values[error],
                    color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                    linewidth=1.35, markersize=4.2, capsize=2.0, label=METHOD_LABELS[method],
                )
            axis.set_xscale("log", base=2)
            ticks = np.sort(subset["n_train"].unique())
            axis.set_xticks(ticks)
            axis.set_xticklabels([str(int(value)) for value in ticks])
            axis.set_xlabel("Training sample size")
            axis.set_ylabel(label)
            if limits[1] is not None:
                axis.set_ylim(*limits)
            axis.set_title(DESIGN_LABELS[design] if column == 0 else "", loc="left", fontsize=9)
            style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.83, wspace=0.28, hspace=0.36)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False)
    figure.suptitle("Conditional and out-of-sample region recovery", y=0.985, fontsize=12)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"section4_3_oos_recovery.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_by_x(by_x: pd.DataFrame, output_dir: Path) -> None:
    """Show query-wise difficulty at the largest training size."""
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=False)
    for axis, design in zip(axes, DESIGNS):
        subset = by_x[by_x["design"] == design]
        for method in ("amqr", "intrinsic_radial", "global_amqr"):
            values = subset[subset["method"] == method].sort_values("query_x")
            axis.errorbar(
                values["query_x"],
                values["jaccard_50_mean"],
                yerr=1.96 * values["jaccard_50_se"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.35,
                markersize=4.2,
                capsize=2.0,
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel("Unseen predictor value x")
        axis.set_ylabel("50% region Jaccard")
        axis.set_ylim(0.0, 1.03)
        axis.set_title(DESIGN_LABELS[design], fontsize=9)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.72, wspace=0.25)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False)
    figure.suptitle("Conditional region recovery across predictor values", y=0.985, fontsize=12)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"section4_3_by_x.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def region_classes(ranks: np.ndarray) -> np.ndarray:
    classes = np.full(len(ranks), 3, dtype=int)
    for class_value, level in reversed(tuple(enumerate((0.25, 0.50, 0.80)))):
        classes[region_mask(ranks, level)] = class_value
    return classes


def plot_branching_examples(examples: Mapping[float, Mapping[str, Any]], output_dir: Path) -> None:
    query_values = sorted(examples)
    methods = ("oracle", "amqr", "intrinsic_radial")
    figure, axes = plt.subplots(3, len(query_values), figsize=(11.7, 8.0), constrained_layout=True)
    colors = np.asarray(["#264653", "#2A9D8F", "#E9C46A", "#D9D9D9"])
    for row, method in enumerate(methods):
        for column, query_x in enumerate(query_values):
            axis = axes[row, column]
            example = examples[query_x]
            test = example["test"]
            branch = np.asarray(test["branch"], dtype=int)
            amplitude = np.asarray(test["amplitude"], dtype=float)
            x_display = amplitude * np.cos(BRANCH_ANGLES[branch])
            y_display = amplitude * np.sin(BRANCH_ANGLES[branch])
            axis.scatter(x_display, y_display, c=colors[region_classes(example["ranks"][method])], s=11, linewidths=0, alpha=0.9)
            for angle in BRANCH_ANGLES:
                axis.plot([0.0, np.cos(angle)], [0.0, np.sin(angle)], color="#777777", linewidth=0.5, alpha=0.35, zorder=0)
            if row == 0:
                axis.set_title(f"x={query_x:.1f}", fontsize=9)
            if column == 0:
                axis.set_ylabel(METHOD_LABELS[method], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
    figure.suptitle("Conditional nested regions for held-out branching responses", fontsize=12)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"section4_3_branching_regions.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "section4_3_conditional_oos")
    parser.add_argument("--sample-sizes", type=parse_int_list, default=(200, 400, 800))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-per-x", type=int, default=300)
    parser.add_argument("--query-grid", type=parse_float_list, default=(0.1, 0.3, 0.5, 0.7, 0.9))
    parser.add_argument("--base-bandwidth", type=float, default=0.14)
    parser.add_argument("--sphere-density-ratio", type=float, default=20.0)
    parser.add_argument("--sphere-oracle-size", type=int, default=600)
    parser.add_argument("--tree-oracle-per-branch", type=int, default=200)
    parser.add_argument("--functional-grid-size", type=int, default=101)
    parser.add_argument("--concentration", type=float, default=12.0)
    parser.add_argument("--uniform-fraction", type=float, default=0.20)
    parser.add_argument("--graph-connections", type=int, default=20)
    parser.add_argument("--interpolation-neighbors", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=20260813)
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        records = pd.read_csv(args.output_dir / "section4_3_records.csv")
        summary = summarize_clustered(records)
        by_x = summarize_by_x(records)
        summary.to_csv(args.output_dir / "section4_3_summary.csv", index=False)
        by_x.to_csv(args.output_dir / "section4_3_by_x.csv", index=False)
        paired_differences(records).to_csv(args.output_dir / "section4_3_paired_differences.csv", index=False)
        plot_recovery(summary, args.output_dir)
        plot_by_x(by_x, args.output_dir)
        print(f"Postprocessed outputs: {args.output_dir}")
        return

    started = time.perf_counter()
    response_grid = np.linspace(0.0, 1.0, args.functional_grid_size)
    oracle: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for design in DESIGNS:
        for query_x in args.query_grid:
            print(f"Preparing oracle: {design}, x={query_x:.1f}", flush=True)
            oracle[(design, float(query_x))] = oracle_support(
                design,
                query_x,
                args.sphere_oracle_size,
                args.tree_oracle_per_branch,
                args.sphere_density_ratio,
                args.concentration,
                args.uniform_fraction,
            )

    maximum_n = max(args.sample_sizes)
    records: list[Dict[str, Any]] = []
    diagnostics: list[Dict[str, Any]] = []
    examples: Dict[float, Mapping[str, Any]] = {}
    for design_index, design in enumerate(DESIGNS):
        true_dimension = 2 if design == "smooth_sphere" else 1
        for repeat in range(args.repeats):
            seed = args.random_state + design_index * 10_000_000 + repeat * 100_000
            rng = np.random.default_rng(seed)
            x_full = rng.uniform(0.0, 1.0, maximum_n)
            if design == "smooth_sphere":
                train_full = sample_sphere(x_full, rng, args.sphere_density_ratio)
            else:
                train_full = sample_tree(
                    x_full, rng, response_grid, args.concentration, args.uniform_fraction
                )
            for n_train in args.sample_sizes:
                print(
                    f"[{design}] repeat {repeat + 1}/{args.repeats}, n={n_train}",
                    flush=True,
                )
                train = {key: np.asarray(value)[:n_train] for key, value in train_full.items()}
                geometry = fit_geometry(train["points"], true_dimension)
                uniform_source = np.full(n_train, 1.0 / n_train)
                global_center = scientific_anchor_index(train, design)
                _, global_scores, global_diagnostics = fit_scores(
                    geometry["intrinsic"], uniform_source, geometry["volume"], global_center
                )
                global_fit = {
                    "source": uniform_source,
                    "center": int(global_center),
                    "scores": global_scores,
                    "runtime_seconds": float(global_diagnostics["runtime_seconds"]),
                }
                bandwidth = args.base_bandwidth * (min(args.sample_sizes) / n_train) ** 0.20
                for query_index, query_x in enumerate(args.query_grid):
                    test_rng = np.random.default_rng(seed + 10_000 + query_index)
                    x_test = np.full(args.test_per_x, query_x)
                    if design == "smooth_sphere":
                        test = sample_sphere(x_test, test_rng, args.sphere_density_ratio)
                    else:
                        test = sample_tree(
                            x_test, test_rng, response_grid, args.concentration, args.uniform_fraction
                        )
                    rows, details = run_query(
                        design,
                        query_x,
                        train,
                        test,
                        oracle[(design, float(query_x))],
                        geometry,
                        bandwidth,
                        args.interpolation_neighbors,
                        args.graph_connections,
                        global_fit,
                    )
                    for row in rows:
                        records.append({
                            "design": design,
                            "design_label": DESIGN_LABELS[design],
                            "repeat": int(repeat),
                            "random_state": int(seed),
                            "n_train": int(n_train),
                            "query_x": float(query_x),
                            "bandwidth": float(bandwidth),
                            **row,
                        })
                    plan_diag = details["amqr_diagnostics"]["plan"]
                    score_diag = details["amqr_diagnostics"]["score"]
                    extension_diag = details["extension_diagnostics"]
                    diagnostics.append({
                        "design": design,
                        "repeat": int(repeat),
                        "n_train": int(n_train),
                        "query_x": float(query_x),
                        "bandwidth": float(bandwidth),
                        "estimated_dimension": float(geometry["dimension"]),
                        "graph_k": int(geometry["graph_diagnostics"]["effective_k_neighbors"]),
                        "volume_k": int(geometry["volume_k"]),
                        "volume_effective_sample_size": float(geometry["volume_diagnostics"]["effective_sample_size"]),
                        "local_effective_sample_size": float(details["local_ess"]),
                        "anchor_pool_size": int(score_diag["anchor_pool_size"]),
                        "anchor_pool_source_mass": float(
                            score_diag["anchor_pool_realized_source_mass"]
                        ),
                        "anchor_pool_radius": float(score_diag["anchor_pool_radius"]),
                        "oos_interpolation_neighbors": int(
                            extension_diag["interpolation_neighbors"]
                        ),
                        "oos_interpolation_bandwidth": float(
                            extension_diag["interpolation_bandwidth"]
                        ),
                        "oos_interpolation_median_effective_size": float(
                            extension_diag["median_interpolation_effective_size"]
                        ),
                        "center_error": float(details["center_error"]),
                        "transport_cost": float(plan_diag["transport_cost"]),
                        "transport_row_error": float(plan_diag["row_marginal_max_error"]),
                        "transport_column_error": float(plan_diag["column_marginal_max_error"]),
                        "solver_converged": bool(plan_diag["solver_converged"]),
                        "fit_runtime_seconds": float(details["amqr_diagnostics"]["runtime_seconds"]),
                    })
                    if (
                        design == "branching_tree"
                        and repeat == args.repeats - 1
                        and n_train == maximum_n
                        and query_x in (args.query_grid[0], args.query_grid[len(args.query_grid) // 2], args.query_grid[-1])
                    ):
                        examples[float(query_x)] = details["example"]

    records_frame = pd.DataFrame(records)
    diagnostics_frame = pd.DataFrame(diagnostics)
    summary = summarize_clustered(records_frame)
    by_x = summarize_by_x(records_frame)
    paired = paired_differences(records_frame)
    records_frame.to_csv(args.output_dir / "section4_3_records.csv", index=False)
    diagnostics_frame.to_csv(args.output_dir / "section4_3_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "section4_3_summary.csv", index=False)
    by_x.to_csv(args.output_dir / "section4_3_by_x.csv", index=False)
    paired.to_csv(args.output_dir / "section4_3_paired_differences.csv", index=False)
    plot_recovery(summary, args.output_dir)
    plot_by_x(by_x, args.output_dir)
    if examples:
        plot_branching_examples(examples, args.output_dir)

    final = summary[(summary["n_train"] == maximum_n) & summary["method"].isin(PRIMARY_METHODS)]
    manifest = {
        "script": str(Path(__file__).resolve()),
        "sample_sizes": list(args.sample_sizes),
        "repeats": int(args.repeats),
        "query_grid": list(args.query_grid),
        "test_per_query": int(args.test_per_x),
        "base_bandwidth": float(args.base_bandwidth),
        "sphere_density_ratio": float(args.sphere_density_ratio),
        "sphere_oracle_size": int(args.sphere_oracle_size),
        "tree_oracle_per_branch": int(args.tree_oracle_per_branch),
        "concentration": float(args.concentration),
        "uniform_fraction": float(args.uniform_fraction),
        "random_state": int(args.random_state),
        "algorithm": "anchor-free conditional intrinsic uniformization followed by post-fit anchor scoring",
        "oracle_only_uses_analytic_geometry": True,
        "runtime_seconds": float(time.perf_counter() - started),
        "final_primary_results": final[
            ["design", "method", "rank_spearman_mean", "rank_mae_mean", "max_coverage_error_mean", "jaccard_50_mean", "mean_region_jaccard_mean", "false_inclusion_50_mean", "rare_branch_recall_80_mean"]
        ].to_dict(orient="records"),
        "all_solvers_converged": bool(diagnostics_frame["solver_converged"].all()),
        "maximum_transport_row_error": float(diagnostics_frame["transport_row_error"].max()),
        "maximum_transport_column_error": float(diagnostics_frame["transport_column_error"].max()),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
