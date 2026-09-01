"""Section 4.2 simulations for Adaptive Manifold Quantile Regions (AMQR).

The script evaluates the current anchor-free AMQR estimator on four supports
whose intrinsic metric and normalized volume are known to the simulator:

* a wavy closed curve;
* the unit sphere;
* a flat product torus;
* a curved rectangular strip with a boundary.

Each support is sampled under two regimes: intrinsic-volume sampling and a
smooth density tilt with maximum-to-minimum ratio 20.  The estimator receives
only ambient pairwise distances.  True intrinsic distances and inverse-density
volume weights are used solely to construct the finite-support oracle and to
evaluate the graph and volume estimators.

The main methods are the current AMQR estimator, an intrinsic-radial ordering,
and a global metric ordering.  Oracle AMQR is a reference target rather than a
competitor.  AMQR with oracle and uniform target weights are retained as
mechanism diagnostics for the density-correction component.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import i0
from scipy.stats import rankdata, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.anchored_uniformization import (
    estimate_intrinsic_dimension,
    estimate_uniform_volume_weights,
    graph_geodesic_distances,
    intrinsic_uniformization_plan,
    pooled_conditional_transport_wasserstein_scores,
    weighted_frechet_medoid,
)
from _paper_simulation_utils import (
    CURVE_LENGTH,
    adaptive_neighbors,
    embed_wavy_curve,
    sphere_geodesic,
    torus_display,
    torus_geodesic,
    wrapped_angle_distance,
)


LEVELS = (0.25, 0.50, 0.80, 0.90)
METHOD_ORDER = (
    "amqr",
    "amqr_shrunk",
    "intrinsic_radial",
    "ambient_radial",
    "global_metric",
    "amqr_oracle_volume",
    "amqr_uniform_target",
)
METHOD_LABELS = {
    "amqr": "AMQR",
    "amqr_shrunk": "AMQR, 50% shrinkage",
    "intrinsic_radial": "Intrinsic radial",
    "ambient_radial": "Ambient radial",
    "global_metric": "Global metric",
    "amqr_oracle_volume": "AMQR + oracle volume",
    "amqr_uniform_target": "AMQR + uniform target",
}
METHOD_COLORS = {
    "amqr": "#D55E00",
    "amqr_shrunk": "#E69F00",
    "intrinsic_radial": "#0072B2",
    "ambient_radial": "#56B4E9",
    "global_metric": "#009E73",
    "amqr_oracle_volume": "#CC79A7",
    "amqr_uniform_target": "#7F7F7F",
}
SCENARIO_LABELS = {
    "wavy_curve": "Wavy curve",
    "sphere": "Sphere",
    "flat_torus": "Flat torus",
    "curved_strip": "Curved strip",
}
REGIME_LABELS = {"uniform": "Uniform sampling", "tilted": "Density ratio 20"}


@dataclass
class Sample:
    scenario: str
    regime: str
    points: np.ndarray
    latent: np.ndarray
    true_distances: np.ndarray
    relative_density: np.ndarray
    true_dimension: int
    plot_points: np.ndarray


def density_beta(density_ratio: float) -> float:
    ratio = float(density_ratio)
    if ratio < 1.0 or not np.isfinite(ratio):
        raise ValueError("density_ratio must be finite and at least one.")
    return 0.5 * math.log(ratio)


def rejection_sample(
    n_samples: int,
    random_state: int,
    candidate_sampler: Callable[[int, np.random.Generator], np.ndarray],
    score_function: Callable[[np.ndarray], np.ndarray],
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    if beta <= 0.0:
        latent = candidate_sampler(int(n_samples), rng)
        return latent, np.ones(int(n_samples), dtype=float)
    accepted: List[np.ndarray] = []
    accepted_count = 0
    while accepted_count < int(n_samples):
        batch = max(1_000, 4 * (int(n_samples) - accepted_count))
        candidate = candidate_sampler(batch, rng)
        score = np.clip(score_function(candidate), -1.0, 1.0)
        keep = rng.random(batch) <= np.exp(beta * (score - 1.0))
        accepted.append(candidate[keep])
        accepted_count += int(np.sum(keep))
    latent = np.concatenate(accepted, axis=0)[: int(n_samples)]
    relative_density = np.exp(beta * np.clip(score_function(latent), -1.0, 1.0))
    return latent, relative_density


def sample_wavy_curve(
    n_samples: int, density_ratio: float, random_state: int
) -> Sample:
    beta = density_beta(density_ratio)

    def candidates(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))

    def score(z: np.ndarray) -> np.ndarray:
        angle = z[:, 0]
        return 0.72 * np.cos(angle - 0.35) + 0.28 * np.cos(2.0 * angle + 0.55)

    latent, density = rejection_sample(
        n_samples, random_state, candidates, score, beta
    )
    angle = latent[:, 0]
    scale = CURVE_LENGTH / (2.0 * np.pi)
    points = embed_wavy_curve(angle)
    return Sample(
        scenario="wavy_curve",
        regime="uniform" if density_ratio == 1.0 else "tilted",
        points=points,
        latent=latent,
        true_distances=scale * wrapped_angle_distance(angle, angle),
        relative_density=density,
        true_dimension=1,
        plot_points=points[:, :2],
    )


def sample_sphere(
    n_samples: int, density_ratio: float, random_state: int
) -> Sample:
    beta = density_beta(density_ratio)

    def candidates(n: int, rng: np.random.Generator) -> np.ndarray:
        values = rng.normal(size=(n, 3))
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    direction = np.asarray([0.64, -0.31, 0.70])
    direction /= np.linalg.norm(direction)

    def score(z: np.ndarray) -> np.ndarray:
        return z @ direction

    points, density = rejection_sample(
        n_samples, random_state, candidates, score, beta
    )
    longitude = np.arctan2(points[:, 1], points[:, 0])
    latitude = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
    return Sample(
        scenario="sphere",
        regime="uniform" if density_ratio == 1.0 else "tilted",
        points=points,
        latent=points,
        true_distances=sphere_geodesic(points),
        relative_density=density,
        true_dimension=2,
        plot_points=np.column_stack([longitude, latitude]),
    )


def sample_flat_torus(
    n_samples: int, density_ratio: float, random_state: int
) -> Sample:
    beta = density_beta(density_ratio)

    def candidates(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 2.0 * np.pi, size=(n, 2))

    def score(z: np.ndarray) -> np.ndarray:
        return 0.58 * np.cos(z[:, 0] - 0.20) + 0.42 * np.cos(z[:, 1] - 4.10)

    latent, density = rejection_sample(
        n_samples, random_state, candidates, score, beta
    )
    points = np.column_stack(
        [
            np.cos(latent[:, 0]),
            np.sin(latent[:, 0]),
            np.cos(latent[:, 1]),
            np.sin(latent[:, 1]),
        ]
    )
    display = torus_display(latent)
    return Sample(
        scenario="flat_torus",
        regime="uniform" if density_ratio == 1.0 else "tilted",
        points=points,
        latent=latent,
        true_distances=torus_geodesic(latent, latent),
        relative_density=density,
        true_dimension=2,
        plot_points=display[:, :2],
    )


def sample_curved_strip(
    n_samples: int, density_ratio: float, random_state: int
) -> Sample:
    beta = density_beta(density_ratio)

    def candidates(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, 2))

    def score(z: np.ndarray) -> np.ndarray:
        return 0.72 * z[:, 0] - 0.28 * np.cos(np.pi * z[:, 1])

    unit, density = rejection_sample(
        n_samples, random_state, candidates, score, beta
    )
    u = 1.35 * unit[:, 0]
    v = 0.70 * unit[:, 1]
    latent = np.column_stack([u, v])
    bend_radius = 1.15
    points = np.column_stack(
        [
            bend_radius * np.cos(u / bend_radius),
            bend_radius * np.sin(u / bend_radius),
            v,
        ]
    )
    return Sample(
        scenario="curved_strip",
        regime="uniform" if density_ratio == 1.0 else "tilted",
        points=points,
        latent=latent,
        true_distances=cdist(latent, latent),
        relative_density=density,
        true_dimension=2,
        plot_points=latent,
    )


GENERATORS: Mapping[str, Callable[[int, float, int], Sample]] = {
    "wavy_curve": sample_wavy_curve,
    "sphere": sample_sphere,
    "flat_torus": sample_flat_torus,
    "curved_strip": sample_curved_strip,
}


def weighted_cdf_levels(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    right = np.searchsorted(sorted_values, values, side="right") - 1
    return np.clip(cumulative[right], 0.0, 1.0)


def reference_anchor_index(sample: Sample) -> Tuple[int, float]:
    """Return the observation nearest a prespecified supported reference state."""
    if sample.scenario == "wavy_curve":
        distance = (CURVE_LENGTH / (2.0 * np.pi)) * wrapped_angle_distance(
            sample.latent[:, 0], np.asarray([0.0])
        )[:, 0]
    elif sample.scenario == "sphere":
        reference = np.asarray([[1.0, 0.0, 0.0]])
        distance = sphere_geodesic(sample.latent, reference)[:, 0]
    elif sample.scenario == "flat_torus":
        reference = np.asarray([[0.0, 0.0]])
        distance = torus_geodesic(sample.latent, reference)[:, 0]
    elif sample.scenario == "curved_strip":
        distance = np.linalg.norm(sample.latent, axis=1)
    else:
        raise KeyError(sample.scenario)
    index = int(np.argmin(distance))
    return index, float(distance[index])


def global_metric_ranks(distances: np.ndarray) -> Tuple[np.ndarray, float]:
    started = time.perf_counter()
    n_samples = len(distances)
    local_ball = rankdata(distances, axis=1, method="average") / n_samples
    score = np.mean(local_ball, axis=0)
    return rankdata(score, method="average") / n_samples, time.perf_counter() - started


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def region_mask(ranks: np.ndarray, level: float) -> np.ndarray:
    values = np.asarray(ranks, dtype=float)
    order = np.argsort(values, kind="mergesort")
    count = max(1, int(round(float(level) * len(values))))
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.asarray(left, dtype=bool) | np.asarray(right, dtype=bool)
    if not np.any(union):
        return 1.0
    return float(np.mean((np.asarray(left, dtype=bool) & np.asarray(right, dtype=bool))[union]))


def fit_amqr(
    distances: np.ndarray,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    anchor_index: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    plan, plan_diag = intrinsic_uniformization_plan(
        distances, source_weights, target_weights, entropic_regularization=0.0
    )
    scores, score_diag = pooled_conditional_transport_wasserstein_scores(
        distances, plan, source_weights, anchor_index
    )
    ranks = weighted_cdf_levels(scores, source_weights)
    return ranks, scores, {
        "runtime_seconds": float(time.perf_counter() - started),
        "transport": plan_diag,
        "scores": score_diag,
    }


def graph_error(
    graph: np.ndarray, truth: np.ndarray, random_state: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(graph)
    count = min(20_000, n * n)
    left = rng.integers(0, n, size=count)
    right = rng.integers(0, n, size=count)
    true_values = truth[left, right]
    graph_values = graph[left, right]
    positive = true_values > 1e-10
    if not np.any(positive):
        return 0.0, 0.0
    scale = float(np.dot(graph_values[positive], true_values[positive]) / np.dot(graph_values[positive], graph_values[positive]))
    relative = np.abs(scale * graph_values[positive] - true_values[positive]) / true_values[positive]
    return float(np.median(relative)), scale


def method_record(
    sample: Sample,
    n_samples: int,
    repeat: int,
    seed: int,
    method: str,
    ranks: np.ndarray,
    oracle_ranks: np.ndarray,
    runtime_seconds: float,
    oracle_center: int,
    method_center: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "scenario": sample.scenario,
        "scenario_label": SCENARIO_LABELS[sample.scenario],
        "regime": sample.regime,
        "regime_label": REGIME_LABELS[sample.regime],
        "n_samples": int(n_samples),
        "repeat": int(repeat),
        "random_state": int(seed),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "rank_spearman": finite_spearman(ranks, oracle_ranks),
        "rank_mae": float(np.mean(np.abs(ranks - oracle_ranks))),
        "center_geodesic_error": float(sample.true_distances[oracle_center, method_center]),
        "runtime_seconds": float(runtime_seconds),
    }
    for level in LEVELS:
        record[f"jaccard_{int(round(100 * level))}"] = jaccard(
            region_mask(ranks, level), region_mask(oracle_ranks, level)
        )
    return record


def run_single(
    scenario: str,
    n_samples: int,
    density_ratio: float,
    repeat: int,
    random_state: int,
    winsor_quantile: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    sample = GENERATORS[scenario](n_samples, density_ratio, random_state)
    local = cdist(sample.points, sample.points)
    dim_k, graph_k, volume_k = adaptive_neighbors(n_samples, sample.true_dimension)
    dimension = estimate_intrinsic_dimension(local, k_neighbors=dim_k)
    graph, graph_diag = graph_geodesic_distances(
        local, k_neighbors=graph_k, ensure_connected=True
    )
    estimated_volume, volume_diag = estimate_uniform_volume_weights(
        local,
        intrinsic_dimension=dimension,
        k_neighbors=volume_k,
        winsor_quantile=winsor_quantile,
    )
    oracle_volume = 1.0 / np.maximum(sample.relative_density, 1e-12)
    oracle_volume /= np.sum(oracle_volume)
    uniform_target = np.full(n_samples, 1.0 / n_samples)
    source = uniform_target.copy()

    anchor_index, anchor_projection_error = reference_anchor_index(sample)
    oracle_medoid, _ = weighted_frechet_medoid(sample.true_distances, source)
    estimated_medoid, _ = weighted_frechet_medoid(graph, source)
    oracle_ranks, oracle_scores, oracle_diag = fit_amqr(
        sample.true_distances, source, oracle_volume, anchor_index
    )
    amqr_ranks, amqr_scores, amqr_diag = fit_amqr(
        graph, source, estimated_volume, anchor_index
    )
    oracle_volume_ranks, _, oracle_volume_diag = fit_amqr(
        graph, source, oracle_volume, anchor_index
    )
    shrunk_volume = 0.50 * uniform_target + 0.50 * estimated_volume
    shrunk_ranks, _, shrunk_diag = fit_amqr(
        graph, source, shrunk_volume, anchor_index
    )
    uniform_ranks, _, uniform_diag = fit_amqr(
        graph, source, uniform_target, anchor_index
    )

    radial_started = time.perf_counter()
    radial_ranks = weighted_cdf_levels(graph[anchor_index], source)
    radial_runtime = time.perf_counter() - radial_started
    ambient_started = time.perf_counter()
    ambient_ranks = weighted_cdf_levels(local[anchor_index], source)
    ambient_runtime = time.perf_counter() - ambient_started
    metric_ranks, metric_runtime = global_metric_ranks(graph)
    metric_center = int(np.argmin(metric_ranks))

    methods = {
        "amqr": (amqr_ranks, anchor_index, amqr_diag["runtime_seconds"]),
        "amqr_shrunk": (
            shrunk_ranks,
            anchor_index,
            shrunk_diag["runtime_seconds"],
        ),
        "intrinsic_radial": (radial_ranks, anchor_index, radial_runtime),
        "ambient_radial": (ambient_ranks, anchor_index, ambient_runtime),
        "global_metric": (metric_ranks, metric_center, metric_runtime),
        "amqr_oracle_volume": (
            oracle_volume_ranks,
            anchor_index,
            oracle_volume_diag["runtime_seconds"],
        ),
        "amqr_uniform_target": (
            uniform_ranks,
            anchor_index,
            uniform_diag["runtime_seconds"],
        ),
    }
    records = [
        method_record(
            sample,
            n_samples,
            repeat,
            random_state,
            method,
            values[0],
            oracle_ranks,
            float(values[2]),
            anchor_index,
            int(values[1]),
        )
        for method, values in methods.items()
    ]

    graph_relative_error, graph_scale = graph_error(
        graph, sample.true_distances, random_state + 700_000
    )
    volume_l1 = float(np.sum(np.abs(estimated_volume - oracle_volume)))
    volume_spearman = (
        float("nan")
        if np.ptp(oracle_volume) <= 1e-14
        else finite_spearman(estimated_volume, oracle_volume)
    )
    diagnostic = {
        "scenario": sample.scenario,
        "regime": sample.regime,
        "n_samples": int(n_samples),
        "repeat": int(repeat),
        "random_state": int(random_state),
        "true_dimension": int(sample.true_dimension),
        "estimated_dimension": float(dimension),
        "dimension_abs_error": float(abs(dimension - sample.true_dimension)),
        "graph_median_relative_error": graph_relative_error,
        "graph_optimal_scale": graph_scale,
        "volume_l1_error": volume_l1,
        "volume_spearman": volume_spearman,
        "estimated_volume_ess": float(1.0 / np.sum(estimated_volume ** 2)),
        "oracle_volume_ess": float(1.0 / np.sum(oracle_volume ** 2)),
        "estimated_center_error": float(
            sample.true_distances[oracle_medoid, estimated_medoid]
        ),
        "anchor_projection_error": anchor_projection_error,
        "graph_k": int(graph_diag["effective_k_neighbors"]),
        "volume_k": int(volume_k),
        "dimension_k": int(dim_k),
        "transport_row_error": float(
            amqr_diag["transport"]["row_marginal_max_error"]
        ),
        "transport_column_error": float(
            amqr_diag["transport"]["column_marginal_max_error"]
        ),
        "anchor_pool_size": int(amqr_diag["scores"]["anchor_pool_size"]),
        "anchor_pool_source_mass": float(
            amqr_diag["scores"]["anchor_pool_realized_source_mass"]
        ),
        "anchor_pool_radius": float(amqr_diag["scores"]["anchor_pool_radius"]),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload = {
        "sample": sample,
        "oracle_ranks": oracle_ranks,
        "oracle_scores": oracle_scores,
        "method_ranks": {method: values[0] for method, values in methods.items()},
        "method_scores": {"amqr": amqr_scores},
        "anchor_index": anchor_index,
        "oracle_medoid": oracle_medoid,
        "estimated_medoid": estimated_medoid,
        "estimated_volume": estimated_volume,
        "oracle_volume": oracle_volume,
        "diagnostic": diagnostic,
        "oracle_transport": oracle_diag,
    }
    return records, diagnostic, payload


def summarize(
    frame: pd.DataFrame, group_columns: Sequence[str], metrics: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else math.nan
            row[f"{metric}_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            row[f"{metric}_se"] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired Monte Carlo contrasts for the primary recovery metrics."""
    comparisons = (
        ("amqr", "intrinsic_radial"),
        ("amqr", "ambient_radial"),
        ("amqr_shrunk", "amqr"),
    )
    rows: List[Dict[str, Any]] = []
    index_columns = ["scenario", "regime", "n_samples", "repeat", "random_state"]
    for left_method, right_method in comparisons:
        left = frame[frame["method"] == left_method].set_index(index_columns)
        right = frame[frame["method"] == right_method].set_index(index_columns)
        common = left.index.intersection(right.index)
        for keys, left_group in left.loc[common].groupby(level=[0, 1, 2]):
            right_group = right.loc[left_group.index]
            for metric in ("rank_mae", "jaccard_25", "jaccard_50", "jaccard_80"):
                difference = (
                    left_group[metric].to_numpy(dtype=float)
                    - right_group[metric].to_numpy(dtype=float)
                )
                standard_error = (
                    float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
                    if len(difference) > 1
                    else 0.0
                )
                rows.append(
                    {
                        "scenario": keys[0],
                        "regime": keys[1],
                        "n_samples": int(keys[2]),
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "mean_difference": float(np.mean(difference)),
                        "standard_error": standard_error,
                        "ci95_lower": float(np.mean(difference) - 1.96 * standard_error),
                        "ci95_upper": float(np.mean(difference) + 1.96 * standard_error),
                        "paired_repeats": int(len(difference)),
                    }
                )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    axis.tick_params(labelsize=8)


def plot_recovery(summary: pd.DataFrame, output_dir: Path) -> None:
    primary = ("amqr", "amqr_shrunk", "intrinsic_radial", "ambient_radial")
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), sharex=True)
    for row, regime in enumerate(("uniform", "tilted")):
        for column, scenario in enumerate(GENERATORS):
            axis = axes[row, column]
            subset = summary[
                (summary["scenario"] == scenario)
                & (summary["regime"] == regime)
            ]
            for method in primary:
                values = subset[subset["method"] == method].sort_values("n_samples")
                axis.errorbar(
                    values["n_samples"],
                    values["jaccard_50_mean"],
                    yerr=1.96 * values["jaccard_50_se"],
                    color=METHOD_COLORS[method],
                    marker={"amqr": "o", "amqr_shrunk": "D", "intrinsic_radial": "s", "ambient_radial": "^"}[method],
                    linewidth=1.4,
                    markersize=4.2,
                    capsize=2.0,
                    label=METHOD_LABELS[method],
                )
            axis.set_xscale("log", base=2)
            ticks = np.sort(subset["n_samples"].unique())
            axis.set_xticks(ticks)
            axis.set_xticklabels([str(int(value)) for value in ticks])
            # Use a row-specific range so that Monte Carlo differences remain
            # visible without implying that the two sampling regimes share the
            # same performance floor.
            axis.set_ylim(0.75 if regime == "uniform" else 0.50, 1.03)
            axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
            if column == 0:
                axis.set_ylabel(
                    f"{REGIME_LABELS[regime]}\n50% region Jaccard", fontsize=9
                )
            if row == 1:
                axis.set_xlabel("Sample size", fontsize=9)
            style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.10, top=0.82, wspace=0.12, hspace=0.20)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Recovery of oracle manifold quantile regions", fontsize=12, y=0.985)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"section4_2_region_recovery.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_error_chain(
    method_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    metrics = (
        ("graph_median_relative_error_mean", "Graph distance relative error"),
        ("volume_l1_error_mean", "Volume-weight L1 error"),
    )
    for axis, (metric, label) in zip(axes[:2], metrics):
        for scenario in GENERATORS:
            values = diagnostic_summary[
                (diagnostic_summary["scenario"] == scenario)
                & (diagnostic_summary["regime"] == "tilted")
            ].sort_values("n_samples")
            axis.plot(
                values["n_samples"],
                values[metric],
                marker="o",
                linewidth=1.3,
                markersize=4,
                label=SCENARIO_LABELS[scenario],
            )
        axis.set_xscale("log", base=2)
        ticks = np.sort(values["n_samples"].unique())
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(int(value)) for value in ticks])
        axis.set_xlabel("Sample size")
        axis.set_ylabel(label)
        style_axis(axis)
    axis = axes[2]
    for scenario in GENERATORS:
        values = method_summary[
            (method_summary["scenario"] == scenario)
            & (method_summary["regime"] == "tilted")
            & (method_summary["method"] == "amqr")
        ].sort_values("n_samples")
        axis.plot(
            values["n_samples"],
            values["rank_mae_mean"],
            marker="o",
            linewidth=1.3,
            markersize=4,
            label=SCENARIO_LABELS[scenario],
        )
    axis.set_xscale("log", base=2)
    ticks = np.sort(values["n_samples"].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(int(value)) for value in ticks])
    axis.set_xlabel("Sample size")
    axis.set_ylabel("AMQR rank MAE")
    style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.76, wspace=0.28)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Estimated error chain under density imbalance", fontsize=12, y=0.985)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"section4_2_error_chain.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_regions(payloads: Mapping[Tuple[str, str], Dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4), constrained_layout=True)
    methods = ("oracle", "amqr")
    for row, method in enumerate(methods):
        for column, scenario in enumerate(GENERATORS):
            axis = axes[row, column]
            payload = payloads[(scenario, "tilted")]
            sample: Sample = payload["sample"]
            ranks = (
                payload["oracle_ranks"]
                if method == "oracle"
                else payload["method_ranks"]["amqr"]
            )
            classes = np.full(len(ranks), 3, dtype=int)
            for class_value, level in reversed(tuple(enumerate((0.25, 0.50, 0.80)))):
                classes[region_mask(ranks, level)] = class_value
            colors = np.asarray(["#264653", "#2A9D8F", "#E9C46A", "#D9D9D9"])
            axis.scatter(
                sample.plot_points[:, 0],
                sample.plot_points[:, 1],
                c=colors[classes],
                s=8,
                linewidths=0,
                alpha=0.85,
                rasterized=True,
            )
            center = payload["anchor_index"]
            axis.scatter(
                sample.plot_points[center, 0],
                sample.plot_points[center, 1],
                marker="*",
                s=100,
                facecolor="#FFFFFF",
                edgecolor="#111111",
                linewidth=0.8,
                zorder=5,
            )
            axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal", adjustable="datalim")
            if column == 0:
                axis.set_ylabel("Oracle" if method == "oracle" else "Estimated AMQR")
    figure.suptitle("Nested regions under nonuniform sampling", fontsize=12)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"section4_2_nested_regions.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def parse_sizes(value: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or min(values) < 30:
        raise argparse.ArgumentTypeError("sample sizes must be at least 30")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=parse_sizes, default=parse_sizes("100,200,400"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--density-ratio", type=float, default=20.0)
    parser.add_argument("--winsor-quantile", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate summaries and figures from existing record CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "section4_2_manifold_regions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        record_frame = pd.read_csv(args.output_dir / "section4_2_records.csv")
        diagnostic_frame = pd.read_csv(args.output_dir / "section4_2_diagnostics.csv")
        method_summary = pd.read_csv(args.output_dir / "section4_2_summary.csv")
        diagnostic_summary = pd.read_csv(
            args.output_dir / "section4_2_diagnostic_summary.csv"
        )
        paired_differences(record_frame).to_csv(
            args.output_dir / "section4_2_paired_differences.csv", index=False
        )
        plot_recovery(method_summary, args.output_dir)
        plot_error_chain(method_summary, diagnostic_summary, args.output_dir)
        display_payloads: Dict[Tuple[str, str], Dict[str, Any]] = {}
        n_samples = max(args.sample_sizes)
        for scenario_index, scenario in enumerate(GENERATORS):
            random_state = (
                args.seed
                + scenario_index * 10_000_000
                + 1_000_000
                + n_samples * 1_000
            )
            _, _, payload = run_single(
                scenario=scenario,
                n_samples=n_samples,
                density_ratio=float(args.density_ratio),
                repeat=0,
                random_state=random_state,
                winsor_quantile=float(args.winsor_quantile),
            )
            display_payloads[(scenario, "tilted")] = payload
        plot_regions(display_payloads, args.output_dir)
        print(f"Postprocessed outputs: {args.output_dir}")
        return
    records: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    display_payloads: Dict[Tuple[str, str], Dict[str, Any]] = {}
    started = time.perf_counter()
    regimes = (("uniform", 1.0), ("tilted", float(args.density_ratio)))
    total = len(GENERATORS) * len(regimes) * len(args.sample_sizes) * args.repeats
    completed = 0
    for scenario_index, scenario in enumerate(GENERATORS):
        for regime_index, (regime, ratio) in enumerate(regimes):
            for n_samples in args.sample_sizes:
                for repeat in range(args.repeats):
                    random_state = (
                        args.seed
                        + scenario_index * 10_000_000
                        + regime_index * 1_000_000
                        + n_samples * 1_000
                        + repeat
                    )
                    completed += 1
                    print(
                        f"[{completed}/{total}] {scenario}, {regime}, n={n_samples}, "
                        f"repeat={repeat + 1}/{args.repeats}",
                        flush=True,
                    )
                    rows, diagnostic, payload = run_single(
                        scenario=scenario,
                        n_samples=n_samples,
                        density_ratio=ratio,
                        repeat=repeat,
                        random_state=random_state,
                        winsor_quantile=float(args.winsor_quantile),
                    )
                    records.extend(rows)
                    diagnostics.append(diagnostic)
                    if n_samples == max(args.sample_sizes) and repeat == 0:
                        display_payloads[(scenario, regime)] = payload

    record_frame = pd.DataFrame(records)
    diagnostic_frame = pd.DataFrame(diagnostics)
    metric_columns = [
        "rank_spearman",
        "rank_mae",
        "jaccard_25",
        "jaccard_50",
        "jaccard_80",
        "jaccard_90",
        "center_geodesic_error",
        "runtime_seconds",
    ]
    diagnostic_columns = [
        "dimension_abs_error",
        "graph_median_relative_error",
        "volume_l1_error",
        "volume_spearman",
        "estimated_volume_ess",
        "oracle_volume_ess",
        "estimated_center_error",
        "anchor_projection_error",
        "transport_row_error",
        "transport_column_error",
        "runtime_seconds",
    ]
    method_summary = summarize(
        record_frame,
        ("scenario", "scenario_label", "regime", "regime_label", "n_samples", "method", "method_label"),
        metric_columns,
    )
    diagnostic_summary = summarize(
        diagnostic_frame,
        ("scenario", "regime", "n_samples"),
        diagnostic_columns,
    )
    record_frame.to_csv(args.output_dir / "section4_2_records.csv", index=False)
    diagnostic_frame.to_csv(args.output_dir / "section4_2_diagnostics.csv", index=False)
    method_summary.to_csv(args.output_dir / "section4_2_summary.csv", index=False)
    diagnostic_summary.to_csv(
        args.output_dir / "section4_2_diagnostic_summary.csv", index=False
    )
    paired_differences(record_frame).to_csv(
        args.output_dir / "section4_2_paired_differences.csv", index=False
    )
    plot_recovery(method_summary, args.output_dir)
    plot_error_chain(method_summary, diagnostic_summary, args.output_dir)
    plot_regions(display_payloads, args.output_dir)

    final_n = max(args.sample_sizes)
    final_primary = method_summary[
        (method_summary["n_samples"] == final_n)
        & method_summary["method"].isin(("amqr", "amqr_shrunk", "intrinsic_radial", "ambient_radial"))
    ][
        [
            "scenario",
            "regime",
            "method",
            "rank_spearman_mean",
            "rank_mae_mean",
            "jaccard_25_mean",
            "jaccard_50_mean",
            "jaccard_80_mean",
        ]
    ]
    manifest = {
        "script": str(Path(__file__).resolve()),
        "sample_sizes": list(args.sample_sizes),
        "repeats": int(args.repeats),
        "density_ratio": float(args.density_ratio),
        "winsor_quantile": float(args.winsor_quantile),
        "seed": int(args.seed),
        "methods": list(METHOD_ORDER),
        "levels": list(LEVELS),
        "runtime_seconds": float(time.perf_counter() - started),
        "final_primary_results": final_primary.to_dict(orient="records"),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nFinal primary summary")
    print(final_primary.to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
