"""Known-geometry Hallin--Liu comparison on the Section 4.2 flat torus.

The Section 4.2 torus density factorises into two von Mises marginals.  Their
population transports to circular uniform distributions are analytic, so the
product map and its pole-centred square-cap ranks form a common population
oracle for every method in this benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, vonmises


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
from models.hallin_liu import signed_wrapped_difference, unconditional_cap_ranks
from run_section4_2_manifold_regions import (
    fit_amqr,
    sample_flat_torus,
    torus_geodesic,
    weighted_cdf_levels,
)
from _paper_simulation_utils import adaptive_neighbors


LEVELS = (0.25, 0.50, 0.80, 0.90)
ANCHOR = np.asarray([0.0, 0.0])
MARGINAL_CENTRES = np.asarray([0.20, 4.10])
SCORE_WEIGHTS = np.asarray([0.58, 0.42])
METHODS = (
    "amqr",
    "hallin_liu",
    "intrinsic_radial",
    "ambient_radial",
    "amqr_oracle_nuisance",
)
METHOD_LABELS = {
    "amqr": "AMQR (distance only)",
    "hallin_liu": "Hallin--Liu (known flat torus)",
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


def population_torus_uniformization(
    angles: np.ndarray, density_ratio: float
) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    beta = 0.5 * math.log(float(density_ratio))
    mapped = np.empty_like(values)
    for coordinate in range(2):
        delta = signed_wrapped_difference(
            values[:, coordinate], MARGINAL_CENTRES[coordinate]
        )
        kappa = beta * SCORE_WEIGHTS[coordinate]
        target_delta = 2.0 * np.pi * vonmises.cdf(delta, kappa) - np.pi
        mapped[:, coordinate] = np.mod(
            MARGINAL_CENTRES[coordinate] + target_delta, 2.0 * np.pi
        )
    return mapped


def population_oracle_ranks(
    angles: np.ndarray, density_ratio: float
) -> np.ndarray:
    mapped = population_torus_uniformization(angles, density_ratio)
    mapped_anchor = population_torus_uniformization(
        ANCHOR[None, :], density_ratio
    )[0]
    difference = np.abs(signed_wrapped_difference(mapped, mapped_anchor[None, :]))
    return np.clip((np.max(difference, axis=1) / np.pi) ** 2, 0.0, 1.0)


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
        truth = np.asarray(oracle) <= level
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
    for keys, group in records.groupby(
        ["regime", "method", "method_label"], sort=True
    ):
        row = dict(zip(("regime", "method", "method_label"), keys))
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
    metrics = ("rank_mae", "jaccard_25", "jaccard_50", "jaccard_80", "jaccard_90")
    comparisons = (
        ("amqr", "hallin_liu"),
        ("amqr", "intrinsic_radial"),
        ("hallin_liu", "intrinsic_radial"),
    )
    rows: List[Dict[str, Any]] = []
    for regime in sorted(records["regime"].unique()):
        subset = records[records["regime"] == regime]
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
                        "regime": regime,
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


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    metrics = (
        ("rank_mae", "Population rank MAE", False),
        ("jaccard_50", "Population 50% region Jaccard", True),
    )
    regimes = ("uniform", "tilted")
    display_methods = METHODS[:4]
    width = 0.19
    x = np.arange(len(regimes), dtype=float)
    for axis, (metric, ylabel, higher_better) in zip(axes, metrics):
        for index, method in enumerate(display_methods):
            values = []
            errors = []
            for regime in regimes:
                row = summary[
                    (summary["regime"] == regime) & (summary["method"] == method)
                ].iloc[0]
                values.append(row[f"{metric}_mean"])
                errors.append(1.96 * row[f"{metric}_se"])
            offset = (index - 1.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xticks(x, ["Uniform", "Density ratio 20"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if higher_better:
            axis.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        "Flat-torus pilot: distance-only AMQR versus known-geometry Hallin--Liu",
        y=1.12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(
        output_dir / "hallin_liu_torus_pilot.png", dpi=220, bbox_inches="tight"
    )
    figure.savefig(output_dir / "hallin_liu_torus_pilot.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--density-ratios", type=parse_float_list, default=(1.0, 20.0)
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=20260815)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "hallin_liu_torus_pilot",
    )
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
            sample = sample_flat_torus(args.n_samples, density_ratio, seed)
            local = cdist(sample.points, sample.points)
            dim_k, graph_k, volume_k = adaptive_neighbors(args.n_samples, 2)
            dimension = estimate_intrinsic_dimension(local, k_neighbors=dim_k)
            graph, graph_diagnostics = graph_geodesic_distances(
                local, k_neighbors=graph_k, ensure_connected=True
            )
            estimated_volume, volume_diagnostics = estimate_uniform_volume_weights(
                local,
                intrinsic_dimension=dimension,
                k_neighbors=volume_k,
                winsor_quantile=args.winsor_quantile,
            )
            source = np.full(args.n_samples, 1.0 / args.n_samples)
            oracle_volume = 1.0 / np.maximum(sample.relative_density, 1e-12)
            oracle_volume /= np.sum(oracle_volume)
            anchor_distance = torus_geodesic(sample.latent, ANCHOR[None, :])[:, 0]
            anchor_index = int(np.argmin(anchor_distance))
            oracle = population_oracle_ranks(sample.latent, density_ratio)

            amqr_ranks, _, amqr_diagnostics = fit_amqr(
                graph, source, estimated_volume, anchor_index
            )
            nuisance_ranks, _, nuisance_diagnostics = fit_amqr(
                sample.true_distances, source, oracle_volume, anchor_index
            )
            hallin_rng = np.random.default_rng(seed + 700_001)
            hallin_ranks, hallin_diagnostics = unconditional_cap_ranks(
                sample.latent,
                anchor_index,
                "flat_torus",
                hallin_rng,
            )

            radial_started = time.perf_counter()
            intrinsic_ranks = weighted_cdf_levels(graph[anchor_index], source)
            intrinsic_runtime = time.perf_counter() - radial_started
            ambient_started = time.perf_counter()
            ambient_ranks = weighted_cdf_levels(local[anchor_index], source)
            ambient_runtime = time.perf_counter() - ambient_started
            fitted = {
                "amqr": (amqr_ranks, amqr_diagnostics["runtime_seconds"]),
                "hallin_liu": (hallin_ranks, hallin_diagnostics["runtime_seconds"]),
                "intrinsic_radial": (intrinsic_ranks, intrinsic_runtime),
                "ambient_radial": (ambient_ranks, ambient_runtime),
                "amqr_oracle_nuisance": (
                    nuisance_ranks,
                    nuisance_diagnostics["runtime_seconds"],
                ),
            }
            rows.extend(
                record(regime, repeat, seed, method, values[0], oracle, values[1])
                for method, values in fitted.items()
            )
            diagnostics.append(
                {
                    "regime": regime,
                    "repeat": int(repeat),
                    "random_state": int(seed),
                    "estimated_dimension": float(dimension),
                    "graph_k": int(graph_diagnostics["effective_k_neighbors"]),
                    "volume_k": int(volume_k),
                    "anchor_projection_error": float(anchor_distance[anchor_index]),
                    "oracle_rank_mean": float(np.mean(oracle)),
                    "oracle_region_50_count": int(np.sum(oracle <= 0.5)),
                    "hallin_liu_n_radial": int(hallin_diagnostics["n_radial"]),
                    "hallin_liu_n_contour": int(hallin_diagnostics["n_contour"]),
                    "hallin_liu_first_mean_cost": float(
                        hallin_diagnostics["first_mean_cost"]
                    ),
                    "hallin_liu_second_mean_cost": float(
                        hallin_diagnostics["second_mean_cost"]
                    ),
                    "amqr_transport_row_error": float(
                        amqr_diagnostics["transport"]["row_marginal_max_error"]
                    ),
                    "amqr_transport_column_error": float(
                        amqr_diagnostics["transport"]["column_marginal_max_error"]
                    ),
                    "volume_weight_l1_error": float(
                        np.sum(np.abs(estimated_volume - oracle_volume))
                    ),
                    "volume_winsor_low": float(volume_diagnostics["winsor_lower"]),
                    "volume_winsor_high": float(volume_diagnostics["winsor_upper"]),
                }
            )

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
        "paper_definition": "two-step pole-centred flat-torus square-cap grid",
        "scientific_anchor": ANCHOR.tolist(),
        "common_oracle": "analytic product von-Mises population OT to flat-torus uniform",
        "n_samples": int(args.n_samples),
        "repeats": int(args.repeats),
        "density_ratios": list(args.density_ratios),
        "random_state": int(args.random_state),
        "runtime_seconds": float(time.perf_counter() - started),
        "all_amqr_solvers_converged": True,
        "primary_summary": summary[
            summary["method"].isin(METHODS[:4])
        ].to_dict(orient="records"),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        summary[
            [
                "regime",
                "method_label",
                "rank_mae_mean",
                "jaccard_50_mean",
                "runtime_seconds_mean",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
