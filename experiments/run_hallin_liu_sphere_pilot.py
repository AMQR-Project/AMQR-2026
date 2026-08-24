"""Pilot comparison with the Hallin--Liu cap-grid construction on S^2.

This is an independent implementation from the definitions in Hallin and Liu
(2024, arXiv:2410.15711); no author software was found.  It implements their
two-step empirical construction for pole-centred spherical-cap regions:

1. transport the sample to an independent uniform grid and map the sampled
   representative of a fixed scientific anchor to an estimated uniform pole;
2. build a structured latitude--longitude grid about that pole and solve a
   second assignment, with ranks determined by latitude layers.

The comparison uses the sphere design from Section 4.2.  Its tilted density is
rotationally symmetric, so the population transport to spherical uniform is
available analytically.  This supplies a common population oracle for every
method and avoids evaluating Hallin--Liu against an AMQR-specific oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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
)
from run_section4_2_manifold_regions import (
    fit_amqr,
    sample_sphere,
    sphere_geodesic,
    weighted_cdf_levels,
)
from _paper_simulation_utils import adaptive_neighbors


LEVELS = (0.25, 0.50, 0.80, 0.90)
ANCHOR = np.asarray([1.0, 0.0, 0.0])
TILT_DIRECTION = np.asarray([0.64, -0.31, 0.70], dtype=float)
TILT_DIRECTION /= np.linalg.norm(TILT_DIRECTION)

METHODS = (
    "amqr",
    "hallin_liu",
    "intrinsic_radial",
    "ambient_radial",
    "amqr_oracle_nuisance",
)
METHOD_LABELS = {
    "amqr": "AMQR (distance only)",
    "hallin_liu": "Hallin--Liu (known sphere)",
    "intrinsic_radial": "Intrinsic radial",
    "ambient_radial": "Ambient radial",
    "amqr_oracle_nuisance": "AMQR + oracle geometry/volume",
}
METHOD_COLORS = {
    "amqr": "#D55E00",
    "hallin_liu": "#7A5195",
    "intrinsic_radial": "#0072B2",
    "ambient_radial": "#56B4E9",
    "amqr_oracle_nuisance": "#CC79A7",
}


def parse_float_list(text: str) -> Tuple[float, ...]:
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def uniform_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(int(n), 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def analytic_spherical_uniformization(
    points: np.ndarray, density_ratio: float
) -> np.ndarray:
    """Population OT for density proportional to exp(beta * direction' y)."""
    values = np.asarray(points, dtype=float)
    beta = 0.5 * math.log(float(density_ratio))
    axial = np.clip(values @ TILT_DIRECTION, -1.0, 1.0)
    if abs(beta) < 1e-12:
        target_axial = axial
    else:
        low = math.exp(-beta)
        target_cdf = (np.exp(beta * axial) - low) / (math.exp(beta) - low)
        target_axial = np.clip(2.0 * target_cdf - 1.0, -1.0, 1.0)

    perpendicular = values - axial[:, None] * TILT_DIRECTION[None, :]
    norm = np.linalg.norm(perpendicular, axis=1)
    directions = np.zeros_like(perpendicular)
    regular = norm > 1e-12
    directions[regular] = perpendicular[regular] / norm[regular, None]
    if np.any(~regular):
        fallback = np.cross(TILT_DIRECTION, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback) < 1e-10:
            fallback = np.cross(TILT_DIRECTION, np.asarray([0.0, 1.0, 0.0]))
        fallback /= np.linalg.norm(fallback)
        directions[~regular] = fallback
    mapped = (
        target_axial[:, None] * TILT_DIRECTION[None, :]
        + np.sqrt(np.maximum(0.0, 1.0 - target_axial**2))[:, None] * directions
    )
    return mapped / np.linalg.norm(mapped, axis=1, keepdims=True)


def population_oracle_ranks(points: np.ndarray, density_ratio: float) -> np.ndarray:
    mapped = analytic_spherical_uniformization(points, density_ratio)
    mapped_anchor = analytic_spherical_uniformization(ANCHOR[None, :], density_ratio)[0]
    # A spherical cap of radius d has uniform probability (1-cos(d))/2.
    return np.clip(0.5 * (1.0 - mapped @ mapped_anchor), 0.0, 1.0)


def tangent_basis(pole: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pole = np.asarray(pole, dtype=float)
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ pole)))]
    first = np.cross(pole, reference)
    first /= np.linalg.norm(first)
    second = np.cross(pole, first)
    second /= np.linalg.norm(second)
    return first, second


def factor_cap_grid(n: int) -> Tuple[int, int, int]:
    """Choose n=n_R*n_S+1 with n_R nearest sqrt(n-1)."""
    remainder = int(n) - 1
    divisors = [value for value in range(2, remainder + 1) if remainder % value == 0]
    if not divisors:
        raise ValueError("The cap pilot requires a composite n-1 (for example n=400).")
    n_r = min(divisors, key=lambda value: abs(value - math.sqrt(remainder)))
    n_s = remainder // n_r
    if n_r > n_s:
        n_r, n_s = n_s, n_r
    return int(n_r), int(n_s), 1


def structured_cap_grid(
    pole: np.ndarray, n_r: int, n_s: int, rotation: float
) -> Tuple[np.ndarray, np.ndarray]:
    first, second = tangent_basis(pole)
    points: List[np.ndarray] = [np.asarray(pole, dtype=float)]
    levels: List[float] = [0.0]
    longitude = rotation + 2.0 * np.pi * np.arange(n_s) / n_s
    tangent = (
        np.cos(longitude)[:, None] * first[None, :]
        + np.sin(longitude)[:, None] * second[None, :]
    )
    for layer in range(1, n_r + 1):
        tau = layer / (n_r + 1.0)
        colatitude = math.acos(1.0 - 2.0 * tau)
        contour = math.cos(colatitude) * pole[None, :] + math.sin(colatitude) * tangent
        points.extend(contour)
        levels.extend([tau] * n_s)
    return np.asarray(points), np.asarray(levels)


def assignment(cost: np.ndarray) -> Tuple[np.ndarray, float]:
    started = time.perf_counter()
    rows, columns = linear_sum_assignment(np.asarray(cost, dtype=float))
    assigned = np.empty(len(rows), dtype=int)
    assigned[rows] = columns
    return assigned, float(time.perf_counter() - started)


def hallin_liu_cap_ranks(
    points: np.ndarray,
    anchor_index: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Two-step Hallin--Liu empirical cap ranks with known S^2 geometry."""
    n = len(points)
    n_r, n_s, n_0 = factor_cap_grid(n)
    preliminary = uniform_sphere(n, rng)
    preliminary_cost = 0.5 * np.arccos(
        np.clip(points @ preliminary.T, -1.0, 1.0)
    ) ** 2
    first_assignment, first_time = assignment(preliminary_cost)
    pole = preliminary[first_assignment[int(anchor_index)]]

    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    regular_grid, grid_levels = structured_cap_grid(pole, n_r, n_s, rotation)
    final_cost = 0.5 * np.arccos(
        np.clip(points @ regular_grid.T, -1.0, 1.0)
    ) ** 2
    second_assignment, second_time = assignment(final_cost)
    ranks = grid_levels[second_assignment]
    return ranks, {
        "n_r": n_r,
        "n_s": n_s,
        "n_0": n_0,
        "pole": pole,
        "first_assignment_seconds": first_time,
        "second_assignment_seconds": second_time,
        "runtime_seconds": first_time + second_time,
        "first_mean_cost": float(np.mean(preliminary_cost[np.arange(n), first_assignment])),
        "second_mean_cost": float(np.mean(final_cost[np.arange(n), second_assignment])),
    }


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    return 1.0 if not np.any(union) else float(np.sum(left & right) / np.sum(union))


def record(
    regime: str,
    repeat: int,
    seed: int,
    method: str,
    ranks: np.ndarray,
    oracle: np.ndarray,
    runtime: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "regime": regime,
        "repeat": int(repeat),
        "random_state": int(seed),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "rank_spearman": finite_spearman(ranks, oracle),
        "rank_mae": float(np.mean(np.abs(np.asarray(ranks) - oracle))),
        "runtime_seconds": float(runtime),
    }
    for level in LEVELS:
        estimated = np.asarray(ranks) <= level + 1e-12
        truth = oracle <= level
        suffix = int(round(100 * level))
        row[f"jaccard_{suffix}"] = jaccard(estimated, truth)
        row[f"coverage_error_{suffix}"] = float(abs(np.mean(estimated) - level))
    return row


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_spearman",
        "rank_mae",
        *(f"jaccard_{int(100 * level)}" for level in LEVELS),
        *(f"coverage_error_{int(100 * level)}" for level in LEVELS),
        "runtime_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in records.groupby(["regime", "method", "method_label"], sort=True):
        row = dict(zip(("regime", "method", "method_label"), keys))
        row["repeats"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(records: pd.DataFrame) -> pd.DataFrame:
    metrics = ("rank_mae", "jaccard_25", "jaccard_50", "jaccard_80", "jaccard_90")
    comparisons = (
        ("amqr", "hallin_liu"),
        ("amqr", "intrinsic_radial"),
        ("hallin_liu", "intrinsic_radial"),
    )
    rows: List[Dict[str, Any]] = []
    for regime in sorted(records["regime"].unique()):
        subset = records[records["regime"] == regime]
        for left, right in comparisons:
            left_data = subset[subset["method"] == left].set_index("repeat")
            right_data = subset[subset["method"] == right].set_index("repeat")
            common = left_data.index.intersection(right_data.index)
            for metric in metrics:
                delta = left_data.loc[common, metric].to_numpy() - right_data.loc[common, metric].to_numpy()
                mean = float(np.mean(delta))
                se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
                rows.append({
                    "regime": regime,
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "mean_difference": mean,
                    "standard_error": se,
                    "ci95_lower": mean - 1.96 * se,
                    "ci95_upper": mean + 1.96 * se,
                    "paired_repeats": int(len(delta)),
                })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    metrics = (("rank_mae", "Population rank MAE", False), ("jaccard_50", "Population 50% region Jaccard", True))
    regimes = ("uniform", "tilted")
    display_methods = METHODS[:4]
    width = 0.19
    x = np.arange(len(regimes), dtype=float)
    for axis, (metric, ylabel, higher_better) in zip(axes, metrics):
        for index, method in enumerate(display_methods):
            values = []
            errors = []
            for regime in regimes:
                row = summary[(summary["regime"] == regime) & (summary["method"] == method)].iloc[0]
                values.append(row[f"{metric}_mean"])
                errors.append(1.96 * row[f"{metric}_se"])
            offset = (index - 1.5) * width
            axis.bar(x + offset, values, width, yerr=errors, capsize=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_xticks(x, ["Uniform", "Density ratio 20"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if higher_better:
            axis.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Sphere pilot: distance-only AMQR versus known-geometry Hallin--Liu", y=1.12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(output_dir / "hallin_liu_sphere_pilot.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "hallin_liu_sphere_pilot.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--density-ratios", type=parse_float_list, default=(1.0, 20.0))
    parser.add_argument("--winsor-quantile", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "hallin_liu_sphere_pilot")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for density_index, density_ratio in enumerate(args.density_ratios):
        regime = "uniform" if float(density_ratio) == 1.0 else "tilted"
        for repeat in range(args.repeats):
            seed = args.random_state + density_index * 1_000_000 + repeat * 10_000
            print(f"[{regime}] repeat {repeat + 1}/{args.repeats}", flush=True)
            sample = sample_sphere(args.n_samples, density_ratio, seed)
            local = cdist(sample.points, sample.points)
            dim_k, graph_k, volume_k = adaptive_neighbors(args.n_samples, 2)
            dimension = estimate_intrinsic_dimension(local, k_neighbors=dim_k)
            graph, graph_diag = graph_geodesic_distances(local, k_neighbors=graph_k, ensure_connected=True)
            estimated_volume, volume_diag = estimate_uniform_volume_weights(
                local,
                intrinsic_dimension=dimension,
                k_neighbors=volume_k,
                winsor_quantile=args.winsor_quantile,
            )
            source = np.full(args.n_samples, 1.0 / args.n_samples)
            oracle_volume = 1.0 / np.maximum(sample.relative_density, 1e-12)
            oracle_volume /= np.sum(oracle_volume)
            anchor_distance = sphere_geodesic(sample.points, ANCHOR[None, :])[:, 0]
            anchor_index = int(np.argmin(anchor_distance))
            oracle = population_oracle_ranks(sample.points, density_ratio)

            amqr_ranks, _, amqr_diag = fit_amqr(graph, source, estimated_volume, anchor_index)
            nuisance_ranks, _, nuisance_diag = fit_amqr(sample.true_distances, source, oracle_volume, anchor_index)
            hallin_rng = np.random.default_rng(seed + 700_001)
            hl_ranks, hl_diag = hallin_liu_cap_ranks(sample.points, anchor_index, hallin_rng)

            radial_started = time.perf_counter()
            intrinsic_ranks = weighted_cdf_levels(graph[anchor_index], source)
            intrinsic_runtime = time.perf_counter() - radial_started
            ambient_started = time.perf_counter()
            ambient_ranks = weighted_cdf_levels(local[anchor_index], source)
            ambient_runtime = time.perf_counter() - ambient_started

            fitted = {
                "amqr": (amqr_ranks, amqr_diag["runtime_seconds"]),
                "hallin_liu": (hl_ranks, hl_diag["runtime_seconds"]),
                "intrinsic_radial": (intrinsic_ranks, intrinsic_runtime),
                "ambient_radial": (ambient_ranks, ambient_runtime),
                "amqr_oracle_nuisance": (nuisance_ranks, nuisance_diag["runtime_seconds"]),
            }
            rows.extend(
                record(regime, repeat, seed, method, values[0], oracle, values[1])
                for method, values in fitted.items()
            )
            diagnostics.append({
                "regime": regime,
                "repeat": repeat,
                "random_state": seed,
                "estimated_dimension": float(dimension),
                "graph_k": int(graph_diag["effective_k_neighbors"]),
                "volume_k": int(volume_k),
                "anchor_projection_error": float(anchor_distance[anchor_index]),
                "oracle_rank_mean": float(np.mean(oracle)),
                "oracle_region_50_count": int(np.sum(oracle <= 0.5)),
                "hallin_liu_n_r": int(hl_diag["n_r"]),
                "hallin_liu_n_s": int(hl_diag["n_s"]),
                "hallin_liu_first_mean_cost": float(hl_diag["first_mean_cost"]),
                "hallin_liu_second_mean_cost": float(hl_diag["second_mean_cost"]),
                "amqr_transport_row_error": float(amqr_diag["transport"]["row_marginal_max_error"]),
                "amqr_transport_column_error": float(amqr_diag["transport"]["column_marginal_max_error"]),
                "volume_weight_l1_error": float(np.sum(np.abs(estimated_volume - oracle_volume))),
                "volume_winsor_low": float(volume_diag["winsor_lower"]),
                "volume_winsor_high": float(volume_diag["winsor_upper"]),
            })

    records = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = summarize(records)
    paired = paired_differences(records)
    records.to_csv(args.output_dir / "records.csv", index=False)
    diagnostic_frame.to_csv(args.output_dir / "diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    plot_summary(summary, args.output_dir)

    manifest = {
        "status": "independent pilot implementation; not author-code reproduction",
        "paper": "Hallin and Liu (2024), arXiv:2410.15711",
        "paper_definition": "two-step pole-centred S^2 cap grid",
        "scientific_anchor": ANCHOR.tolist(),
        "common_oracle": "analytic rotationally symmetric population OT to spherical uniform",
        "n_samples": int(args.n_samples),
        "repeats": int(args.repeats),
        "density_ratios": list(args.density_ratios),
        "random_state": int(args.random_state),
        "runtime_seconds": float(time.perf_counter() - started),
        "primary_summary": summary[summary["method"].isin(METHODS[:4])].to_dict(orient="records"),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[["regime", "method_label", "rank_mae_mean", "jaccard_50_mean", "runtime_seconds_mean"]].to_string(index=False))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
