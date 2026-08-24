"""Conditional sphere benchmark against known-geometry Hallin--Liu grids.

The conditional density is the smooth moving von Mises--Fisher tilt used in
Section 4.3.  Its population transport to spherical uniform is analytic at
every query, giving a common population oracle.  Hallin--Liu is implemented
with the known spherical metric and a two-step conditional Kantorovich grid.
For held-out responses, its expected target-layer scores are interpolated and
calibrated by the same local rule used for the AMQR comparison.  This scalar
OOS step is an explicitly documented benchmark extension, not author code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.anchored_uniformization import extend_anchored_ranks
from models.hallin_liu import conditional_sphere_layer_scores
from _paper_simulation_utils import sphere_geodesic
from run_section4_3_conditional_oos import (
    SPHERE_REFERENCE,
    fit_geometry,
    fit_scores,
    interpolate_raw_scores,
    kernel_weights,
    sample_sphere,
    sphere_direction,
    weighted_cdf_at,
)


LEVELS = (0.25, 0.50, 0.80, 0.90)
METHODS = (
    "amqr",
    "hallin_liu",
    "intrinsic_radial",
    "ambient_radial",
    "global_amqr",
)
METHOD_LABELS = {
    "amqr": "Conditional AMQR (distance only)",
    "hallin_liu": "Hallin-Liu + common OOS extension",
    "intrinsic_radial": "Conditional intrinsic radial",
    "ambient_radial": "Conditional ambient radial",
    "global_amqr": "Unconditional AMQR",
}
METHOD_COLORS = {
    "amqr": "#D55E00",
    "hallin_liu": "#7A5195",
    "intrinsic_radial": "#0072B2",
    "ambient_radial": "#56B4E9",
    "global_amqr": "#009E73",
}


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


def analytic_spherical_uniformization(
    points: np.ndarray, direction: np.ndarray, density_ratio: float
) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    beta = 0.5 * math.log(float(density_ratio))
    axial = np.clip(values @ direction, -1.0, 1.0)
    if abs(beta) < 1e-12:
        target_axial = axial
    else:
        low = math.exp(-beta)
        target_cdf = (np.exp(beta * axial) - low) / (math.exp(beta) - low)
        target_axial = np.clip(2.0 * target_cdf - 1.0, -1.0, 1.0)
    perpendicular = values - axial[:, None] * direction[None, :]
    norm = np.linalg.norm(perpendicular, axis=1)
    tangent = np.zeros_like(perpendicular)
    regular = norm > 1e-12
    tangent[regular] = perpendicular[regular] / norm[regular, None]
    if np.any(~regular):
        fallback = np.cross(direction, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback) < 1e-10:
            fallback = np.cross(direction, np.asarray([0.0, 1.0, 0.0]))
        fallback /= np.linalg.norm(fallback)
        tangent[~regular] = fallback
    mapped = (
        target_axial[:, None] * direction[None, :]
        + np.sqrt(np.maximum(0.0, 1.0 - target_axial**2))[:, None] * tangent
    )
    return mapped / np.linalg.norm(mapped, axis=1, keepdims=True)


def population_oracle_ranks(
    points: np.ndarray, query_x: float, density_ratio: float
) -> np.ndarray:
    direction = sphere_direction(np.asarray([query_x]))[0]
    mapped = analytic_spherical_uniformization(points, direction, density_ratio)
    mapped_anchor = analytic_spherical_uniformization(
        SPHERE_REFERENCE[None, :], direction, density_ratio
    )[0]
    return np.clip(0.5 * (1.0 - mapped @ mapped_anchor), 0.0, 1.0)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    return 1.0 if not np.any(union) else float(np.sum(left & right) / np.sum(union))


def evaluate(
    method: str,
    ranks: np.ndarray,
    oracle: np.ndarray,
    local_effective_sample_size: float,
    runtime_seconds: float,
) -> Dict[str, Any]:
    values = np.asarray(ranks, dtype=float)
    truth_ranks = np.asarray(oracle, dtype=float)
    record: Dict[str, Any] = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "rank_spearman": finite_spearman(values, truth_ranks),
        "rank_mae": float(np.mean(np.abs(values - truth_ranks))),
        "local_effective_sample_size": float(local_effective_sample_size),
        "runtime_seconds": float(runtime_seconds),
    }
    overlaps: List[float] = []
    coverage_errors: List[float] = []
    for level in LEVELS:
        estimated = values <= level
        truth = truth_ranks <= level
        suffix = int(round(100 * level))
        record[f"jaccard_{suffix}"] = jaccard(estimated, truth)
        record[f"coverage_error_{suffix}"] = float(abs(np.mean(estimated) - level))
        overlaps.append(record[f"jaccard_{suffix}"])
        coverage_errors.append(record[f"coverage_error_{suffix}"])
    record["mean_region_jaccard"] = float(np.mean(overlaps))
    record["mean_coverage_error"] = float(np.mean(coverage_errors))
    record["max_coverage_error"] = float(np.max(coverage_errors))
    record["false_inclusion_50"] = float(
        np.mean((values <= 0.50) & ~(truth_ranks <= 0.50))
    )
    return record


def summarize_clustered(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_spearman",
        "rank_mae",
        "mean_coverage_error",
        "max_coverage_error",
        "mean_region_jaccard",
        "jaccard_25",
        "jaccard_50",
        "jaccard_80",
        "jaccard_90",
        "false_inclusion_50",
        "local_effective_sample_size",
        "runtime_seconds",
    ]
    per_repeat = records.groupby(
        ["n_train", "method", "method_label", "repeat"], as_index=False
    )[metrics].mean()
    rows: List[Dict[str, Any]] = []
    for keys, group in per_repeat.groupby(
        ["n_train", "method", "method_label"], sort=True
    ):
        row = dict(zip(("n_train", "method", "method_label"), keys))
        row["repeats"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_se"] = float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(records: pd.DataFrame) -> pd.DataFrame:
    metrics = ("rank_mae", "max_coverage_error", "jaccard_50", "mean_region_jaccard")
    comparisons = (
        ("amqr", "hallin_liu"),
        ("amqr", "intrinsic_radial"),
        ("amqr", "global_amqr"),
        ("hallin_liu", "intrinsic_radial"),
    )
    per_repeat = records.groupby(
        ["n_train", "repeat", "method"], as_index=False
    )[list(metrics)].mean()
    rows: List[Dict[str, Any]] = []
    for n_train in sorted(records["n_train"].unique()):
        subset = per_repeat[per_repeat["n_train"] == n_train]
        for left_method, right_method in comparisons:
            left = subset[subset["method"] == left_method].set_index("repeat")
            right = subset[subset["method"] == right_method].set_index("repeat")
            common = left.index.intersection(right.index)
            for metric in metrics:
                difference = (
                    left.loc[common, metric].to_numpy(float)
                    - right.loc[common, metric].to_numpy(float)
                )
                mean = float(np.mean(difference))
                standard_error = float(
                    np.std(difference, ddof=1) / np.sqrt(len(difference))
                )
                rows.append(
                    {
                        "n_train": int(n_train),
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "mean_difference": mean,
                        "standard_error": standard_error,
                        "ci95_lower": mean - 1.96 * standard_error,
                        "ci95_upper": mean + 1.96 * standard_error,
                        "paired_repeats": int(len(difference)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_by_x(records: pd.DataFrame) -> pd.DataFrame:
    maximum_n = int(records["n_train"].max())
    selected = records[records["n_train"] == maximum_n]
    rows: List[Dict[str, Any]] = []
    for keys, group in selected.groupby(["query_x", "method"], sort=True):
        row = dict(zip(("query_x", "method"), keys))
        row["n_train"] = maximum_n
        row["repeats"] = int(len(group))
        for metric in ("rank_mae", "jaccard_50", "max_coverage_error"):
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_se"] = float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recovery(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    configurations = (
        ("rank_mae", "Population rank MAE"),
        ("jaccard_50", "Population 50% region Jaccard"),
    )
    for axis, (metric, ylabel) in zip(axes, configurations):
        for method in METHODS:
            values = summary[summary["method"] == method].sort_values("n_train")
            axis.errorbar(
                values["n_train"],
                values[f"{metric}_mean"],
                yerr=1.96 * values[f"{metric}_se"],
                marker="o",
                linewidth=1.7,
                capsize=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel("Training sample size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if metric == "jaccard_50":
            axis.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.suptitle(
        "Conditional sphere: distance-only AMQR versus known-geometry Hallin-Liu",
        y=1.13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(
        output_dir / "hallin_liu_conditional_sphere.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "hallin_liu_conditional_sphere.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "hallin_liu_conditional_sphere",
    )
    parser.add_argument("--sample-sizes", type=parse_int_list, default=(200, 400, 800))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-per-x", type=int, default=300)
    parser.add_argument(
        "--query-grid",
        type=parse_float_list,
        default=(0.1, 0.3, 0.5, 0.7, 0.9),
    )
    parser.add_argument("--base-bandwidth", type=float, default=0.14)
    parser.add_argument("--sphere-density-ratio", type=float, default=20.0)
    parser.add_argument("--graph-connections", type=int, default=20)
    parser.add_argument("--interpolation-neighbors", type=int, default=15)
    parser.add_argument("--hallin-radial-layers", type=int, default=20)
    parser.add_argument("--hallin-contour-points", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    maximum_n = max(args.sample_sizes)
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        seed = args.random_state + repeat * 100_000
        rng = np.random.default_rng(seed)
        x_full = rng.uniform(0.0, 1.0, maximum_n)
        train_full = sample_sphere(x_full, rng, args.sphere_density_ratio)
        for n_train in args.sample_sizes:
            print(
                f"[conditional sphere] repeat {repeat + 1}/{args.repeats}, n={n_train}",
                flush=True,
            )
            train = {
                key: np.asarray(value)[:n_train] for key, value in train_full.items()
            }
            geometry = fit_geometry(train["points"], 2)
            intrinsic = np.asarray(geometry["intrinsic"])
            center = int(
                np.argmin(
                    np.linalg.norm(
                        np.asarray(train["points"]) - SPHERE_REFERENCE[None, :],
                        axis=1,
                    )
                )
            )
            bandwidth = args.base_bandwidth * (min(args.sample_sizes) / n_train) ** 0.20
            global_source = np.full(n_train, 1.0 / n_train)
            _, global_scores, global_diagnostics = fit_scores(
                intrinsic, global_source, geometry["volume"], center
            )
            for query_index, query_x in enumerate(args.query_grid):
                source = kernel_weights(train["x"], query_x, bandwidth)
                local_effective_sample_size = float(1.0 / np.sum(source**2))
                effective_interpolation_neighbors = max(
                    int(args.interpolation_neighbors),
                    int(np.ceil(np.sqrt(local_effective_sample_size))),
                )
                amqr_train_ranks, amqr_scores, amqr_diagnostics = fit_scores(
                    intrinsic, source, geometry["volume"], center
                )
                hallin_rng = np.random.default_rng(
                    seed + 5_000_000 + n_train * 100 + query_index
                )
                hallin_scores, hallin_diagnostics = conditional_sphere_layer_scores(
                    train["points"],
                    source,
                    center,
                    hallin_rng,
                    n_radial=args.hallin_radial_layers,
                    n_contour=args.hallin_contour_points,
                )

                test_rng = np.random.default_rng(seed + 10_000 + query_index)
                x_test = np.full(args.test_per_x, query_x)
                test = sample_sphere(x_test, test_rng, args.sphere_density_ratio)
                ambient_cross = cdist(test["points"], train["points"])
                amqr_oos, _, graph_cross, extension_diagnostics = extend_anchored_ranks(
                    ambient_cross,
                    intrinsic,
                    amqr_scores,
                    source,
                    graph_distance_scale=float(
                        geometry["graph_diagnostics"][
                            "median_graph_distance_before_scaling"
                        ]
                    ),
                    graph_connections=args.graph_connections,
                    interpolation_neighbors=effective_interpolation_neighbors,
                )
                exact_cross = sphere_geodesic(test["points"], train["points"])
                hallin_oos, _ = interpolate_raw_scores(
                    exact_cross,
                    hallin_scores,
                    source,
                    args.interpolation_neighbors,
                )
                intrinsic_radial = weighted_cdf_at(
                    intrinsic[center], source, graph_cross[:, center]
                )
                ambient_radial = weighted_cdf_at(
                    geometry["local"][center], source, ambient_cross[:, center]
                )
                global_oos, _ = interpolate_raw_scores(
                    graph_cross,
                    global_scores,
                    global_source,
                    args.interpolation_neighbors,
                )
                oracle = population_oracle_ranks(
                    test["points"], query_x, args.sphere_density_ratio
                )
                method_values = {
                    "amqr": (
                        amqr_oos,
                        amqr_diagnostics["runtime_seconds"]
                        + extension_diagnostics["runtime_seconds"],
                    ),
                    "hallin_liu": (
                        hallin_oos,
                        hallin_diagnostics["runtime_seconds"],
                    ),
                    "intrinsic_radial": (
                        intrinsic_radial,
                        extension_diagnostics["runtime_seconds"],
                    ),
                    "ambient_radial": (ambient_radial, 0.0),
                    "global_amqr": (
                        global_oos,
                        global_diagnostics["runtime_seconds"],
                    ),
                }
                for method, values in method_values.items():
                    rows.append(
                        {
                            "repeat": int(repeat),
                            "random_state": int(seed),
                            "n_train": int(n_train),
                            "query_x": float(query_x),
                            "bandwidth": float(bandwidth),
                            **evaluate(
                                method,
                                values[0],
                                oracle,
                                local_effective_sample_size,
                                float(values[1]),
                            ),
                        }
                    )
                diagnostics.append(
                    {
                        "repeat": int(repeat),
                        "random_state": int(seed),
                        "n_train": int(n_train),
                        "query_x": float(query_x),
                        "bandwidth": float(bandwidth),
                        "local_effective_sample_size": local_effective_sample_size,
                        "oracle_rank_mean": float(np.mean(oracle)),
                        "oracle_region_50_fraction": float(np.mean(oracle <= 0.50)),
                        "hallin_grid_size": int(hallin_diagnostics["grid_size"]),
                        "hallin_first_transport_cost": float(
                            hallin_diagnostics["first_transport_cost"]
                        ),
                        "hallin_second_transport_cost": float(
                            hallin_diagnostics["second_transport_cost"]
                        ),
                        "hallin_row_marginal_max_error": float(
                            hallin_diagnostics["row_marginal_max_error"]
                        ),
                        "hallin_column_marginal_max_error": float(
                            hallin_diagnostics["column_marginal_max_error"]
                        ),
                        "hallin_solver_converged": bool(
                            hallin_diagnostics["solver_converged"]
                        ),
                        "amqr_row_marginal_max_error": float(
                            amqr_diagnostics["plan"]["row_marginal_max_error"]
                        ),
                        "amqr_column_marginal_max_error": float(
                            amqr_diagnostics["plan"]["column_marginal_max_error"]
                        ),
                        "amqr_solver_converged": bool(
                            amqr_diagnostics["plan"]["solver_converged"]
                        ),
                    }
                )

    records = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = summarize_clustered(records)
    paired = paired_differences(records)
    by_x = summarize_by_x(records)
    records.to_csv(args.output_dir / "records.csv", index=False)
    diagnostic_frame.to_csv(args.output_dir / "diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    by_x.to_csv(args.output_dir / "by_x.csv", index=False)
    plot_recovery(summary, args.output_dir)
    manifest = {
        "status": "independent benchmark implementation; not author-code reproduction",
        "paper": "Hallin and Liu (2024), arXiv:2410.15711",
        "known_geometry_baseline": "two-step conditional spherical cap grid",
        "oos_extension": "source-conditional expected grid layer, 15-neighbour geodesic interpolation, weighted empirical-CDF calibration",
        "common_oracle": "analytic moving von-Mises--Fisher population OT to spherical uniform",
        "sample_sizes": list(args.sample_sizes),
        "repeats": int(args.repeats),
        "query_grid": list(args.query_grid),
        "test_per_query": int(args.test_per_x),
        "base_bandwidth": float(args.base_bandwidth),
        "sphere_density_ratio": float(args.sphere_density_ratio),
        "hallin_grid_size": int(
            args.hallin_radial_layers * args.hallin_contour_points + 1
        ),
        "random_state": int(args.random_state),
        "runtime_seconds": float(time.perf_counter() - started),
        "all_hallin_solvers_converged": bool(
            diagnostic_frame["hallin_solver_converged"].all()
        ),
        "all_amqr_solvers_converged": bool(
            diagnostic_frame["amqr_solver_converged"].all()
        ),
        "maximum_hallin_marginal_error": float(
            max(
                diagnostic_frame["hallin_row_marginal_max_error"].max(),
                diagnostic_frame["hallin_column_marginal_max_error"].max(),
            )
        ),
        "maximum_amqr_marginal_error": float(
            max(
                diagnostic_frame["amqr_row_marginal_max_error"].max(),
                diagnostic_frame["amqr_column_marginal_max_error"].max(),
            )
        ),
        "final_results": summary[
            summary["n_train"] == max(args.sample_sizes)
        ].to_dict(orient="records"),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        summary[
            [
                "n_train",
                "method_label",
                "rank_mae_mean",
                "jaccard_50_mean",
                "max_coverage_error_mean",
                "runtime_seconds_mean",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
