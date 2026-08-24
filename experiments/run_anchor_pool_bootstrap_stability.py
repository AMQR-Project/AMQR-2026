"""Anchor-pool and weighted-bootstrap audits for current anchor-indexed AMQR.

Part I varies only the local anchor-pool mass while keeping the fitted
anchor-free transport plan fixed.  Each choice is compared with both the
primary default finite-support oracle and a matched oracle using the same pool.

Part II performs a Bayesian bootstrap on source weights over the fixed observed
support.  Geometry, estimated intrinsic-volume weights, scientific anchor, and
the default n^{-1/2} pool-mass tuning are held fixed; the transport plan and
rank calibration are refitted.  This isolates transport/ranking/region
sensitivity to empirical-law perturbations.  It is not a full support-resample
bootstrap and does not include graph or volume-estimation uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import t


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
)
from run_section4_2_manifold_regions import (
    GENERATORS,
    LEVELS,
    SCENARIO_LABELS,
    adaptive_neighbors,
    finite_spearman,
    jaccard,
    reference_anchor_index,
    weighted_cdf_levels,
)


DEFAULT_SCENARIOS = ("sphere", "flat_torus", "curved_strip")
POOL_ORDER = ("single_row", "half_default", "default", "double_default")
POOL_LABELS = {
    "single_row": "Single row",
    "half_default": r"$0.5n^{-1/2}$",
    "default": r"$n^{-1/2}$",
    "double_default": r"$2n^{-1/2}$",
}
COLORS = {
    "sphere": "#0072B2",
    "flat_torus": "#009E73",
    "curved_strip": "#D55E00",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def mean_ci(values: Iterable[float]) -> Tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = int(len(array))
    if count == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(array))
    if count == 1:
        return mean, math.nan, math.nan, count
    half_width = float(
        t.ppf(0.975, count - 1) * np.std(array, ddof=1) / np.sqrt(count)
    )
    return mean, mean - half_width, mean + half_width, count


def summarize(
    frame: pd.DataFrame, group_columns: Sequence[str], metric_columns: Sequence[str]
) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_columns, key))
        for metric in metric_columns:
            mean, lower, upper, count = mean_ci(group[metric])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_lower"] = lower
            record[f"{metric}_ci_upper"] = upper
            record[f"{metric}_n"] = count
        output.append(record)
    return pd.DataFrame(output)


def pool_specifications(n_samples: int) -> Mapping[str, Tuple[float, int]]:
    default_mass = float(n_samples ** -0.5)
    return {
        "single_row": (1.0 / n_samples, 1),
        "half_default": (0.5 * default_mass, 3),
        "default": (default_mass, 3),
        "double_default": (min(1.0, 2.0 * default_mass), 3),
    }


def score_plan(
    distances: np.ndarray,
    plan: np.ndarray,
    source_weights: np.ndarray,
    anchor_index: int,
    pool_mass: float,
    minimum_rows: int,
    mass_tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores, diagnostics = pooled_conditional_transport_wasserstein_scores(
        distances,
        plan,
        source_weights,
        anchor_index,
        anchor_pool_mass=float(pool_mass),
        minimum_anchor_rows=int(minimum_rows),
        mass_tolerance=float(mass_tolerance),
    )
    return weighted_cdf_levels(scores, source_weights), diagnostics


def threshold_mask(ranks: np.ndarray, level: float) -> np.ndarray:
    return np.asarray(ranks, dtype=float) <= float(level) + 1e-12


def evaluate_ranks(
    ranks: np.ndarray,
    reference: np.ndarray,
    *,
    prefix: str,
    threshold_regions: bool = False,
) -> Dict[str, float]:
    output = {
        f"{prefix}rank_spearman": finite_spearman(ranks, reference),
        f"{prefix}rank_mae": float(np.mean(np.abs(ranks - reference))),
    }
    for level in LEVELS:
        label = int(round(100 * level))
        if threshold_regions:
            left = threshold_mask(ranks, level)
            right = threshold_mask(reference, level)
        else:
            order_left = np.argsort(ranks, kind="mergesort")
            order_right = np.argsort(reference, kind="mergesort")
            count = max(1, int(round(level * len(ranks))))
            left = np.zeros(len(ranks), dtype=bool)
            right = np.zeros(len(ranks), dtype=bool)
            left[order_left[:count]] = True
            right[order_right[:count]] = True
        output[f"{prefix}jaccard_{label}"] = jaccard(left, right)
    return output


def prepare_dataset(
    scenario: str,
    n_samples: int,
    density_ratio: float,
    random_state: int,
    winsor_quantile: float,
) -> Dict[str, Any]:
    sample = GENERATORS[scenario](n_samples, density_ratio, random_state)
    local = cdist(sample.points, sample.points)
    dim_k, graph_k, volume_k = adaptive_neighbors(n_samples, sample.true_dimension)
    dimension = estimate_intrinsic_dimension(local, k_neighbors=dim_k)
    graph, graph_diagnostics = graph_geodesic_distances(
        local, k_neighbors=graph_k, ensure_connected=True
    )
    target, volume_diagnostics = estimate_uniform_volume_weights(
        local,
        intrinsic_dimension=dimension,
        k_neighbors=volume_k,
        winsor_quantile=winsor_quantile,
    )
    oracle_target = 1.0 / np.maximum(sample.relative_density, 1e-12)
    oracle_target /= np.sum(oracle_target)
    source = np.full(n_samples, 1.0 / n_samples)
    anchor_index, anchor_projection_error = reference_anchor_index(sample)
    estimated_plan, estimated_transport = intrinsic_uniformization_plan(
        graph, source, target, entropic_regularization=0.0
    )
    oracle_plan, oracle_transport = intrinsic_uniformization_plan(
        sample.true_distances, source, oracle_target, entropic_regularization=0.0
    )
    return {
        "sample": sample,
        "local": local,
        "graph": graph,
        "source": source,
        "target": target,
        "oracle_target": oracle_target,
        "anchor_index": anchor_index,
        "anchor_projection_error": anchor_projection_error,
        "estimated_plan": estimated_plan,
        "oracle_plan": oracle_plan,
        "estimated_dimension": dimension,
        "graph_diagnostics": graph_diagnostics,
        "volume_diagnostics": volume_diagnostics,
        "estimated_transport": estimated_transport,
        "oracle_transport": oracle_transport,
    }


def pool_audit(
    prepared: Dict[str, Any],
    repeat: int,
    random_state: int,
    mass_tolerance: float,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    sample = prepared["sample"]
    source = prepared["source"]
    anchor_index = prepared["anchor_index"]
    specifications = pool_specifications(len(source))
    estimated: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    oracle: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for name in POOL_ORDER:
        mass, minimum_rows = specifications[name]
        estimated[name] = score_plan(
            prepared["graph"],
            prepared["estimated_plan"],
            source,
            anchor_index,
            mass,
            minimum_rows,
            mass_tolerance,
        )
        oracle[name] = score_plan(
            sample.true_distances,
            prepared["oracle_plan"],
            source,
            anchor_index,
            mass,
            minimum_rows,
            mass_tolerance,
        )

    default_ranks = estimated["default"][0]
    primary_oracle = oracle["default"][0]
    records: List[Dict[str, Any]] = []
    for name in POOL_ORDER:
        ranks, score = estimated[name]
        matched_oracle, oracle_score = oracle[name]
        requested_mass, minimum_rows = specifications[name]
        record: Dict[str, Any] = {
            "scenario": sample.scenario,
            "scenario_label": SCENARIO_LABELS[sample.scenario],
            "regime": sample.regime,
            "n_samples": int(len(source)),
            "repeat": int(repeat),
            "random_state": int(random_state),
            "pool_variant": name,
            "pool_label": POOL_LABELS[name],
            "pool_requested_mass": float(requested_mass),
            "minimum_anchor_rows": int(minimum_rows),
            "pool_realized_mass": score["anchor_pool_realized_source_mass"],
            "pool_size": score["anchor_pool_size"],
            "pool_radius": score["anchor_pool_radius"],
            "anchor_conditional_support_size": score[
                "anchor_conditional_support_size"
            ],
            "oracle_pool_size": oracle_score["anchor_pool_size"],
            "oracle_pool_radius": oracle_score["anchor_pool_radius"],
            "score_runtime_seconds": score["runtime_seconds"],
        }
        record.update(
            evaluate_ranks(
                ranks, primary_oracle, prefix="to_primary_oracle_"
            )
        )
        record.update(
            evaluate_ranks(
                ranks, matched_oracle, prefix="to_matched_oracle_"
            )
        )
        record.update(evaluate_ranks(ranks, default_ranks, prefix="vs_default_"))
        records.append(record)
    return records, default_ranks, {
        "primary_oracle_ranks": primary_oracle,
        "pool_specifications": specifications,
    }


def bootstrap_audit(
    prepared: Dict[str, Any],
    base_ranks: np.ndarray,
    repeat: int,
    random_state: int,
    bootstrap_replicates: int,
    dirichlet_alpha: float,
    mass_tolerance: float,
) -> List[Dict[str, Any]]:
    sample = prepared["sample"]
    n_samples = len(base_ranks)
    anchor_index = prepared["anchor_index"]
    fixed_pool_mass = float(n_samples ** -0.5)
    rng = np.random.default_rng(random_state + 7_919)
    records: List[Dict[str, Any]] = []
    for bootstrap in range(bootstrap_replicates):
        started = time.perf_counter()
        source = rng.dirichlet(np.full(n_samples, float(dirichlet_alpha)))
        record: Dict[str, Any] = {
            "scenario": sample.scenario,
            "scenario_label": SCENARIO_LABELS[sample.scenario],
            "regime": sample.regime,
            "n_samples": int(n_samples),
            "repeat": int(repeat),
            "random_state": int(random_state),
            "bootstrap": int(bootstrap),
            "bootstrap_seed": int(random_state + 7_919),
            "dirichlet_alpha": float(dirichlet_alpha),
            "source_effective_sample_size": float(1.0 / np.sum(source ** 2)),
            "fixed_pool_mass": fixed_pool_mass,
            "status": "ok",
            "error": "",
        }
        try:
            plan, transport = intrinsic_uniformization_plan(
                prepared["graph"],
                source,
                prepared["target"],
                entropic_regularization=0.0,
            )
            ranks, score = score_plan(
                prepared["graph"],
                plan,
                source,
                anchor_index,
                fixed_pool_mass,
                3,
                mass_tolerance,
            )
            record.update(
                evaluate_ranks(
                    ranks,
                    base_ranks,
                    prefix="vs_base_",
                    threshold_regions=True,
                )
            )
            for level in LEVELS:
                label = int(round(100 * level))
                mask = threshold_mask(ranks, level)
                record[f"region_{label}_base_support_fraction"] = float(
                    np.mean(mask)
                )
                record[f"region_{label}_bootstrap_mass"] = float(
                    np.sum(source[mask])
                )
            record.update(
                {
                    "pool_realized_mass": score[
                        "anchor_pool_realized_source_mass"
                    ],
                    "pool_size": score["anchor_pool_size"],
                    "pool_radius": score["anchor_pool_radius"],
                    "row_marginal_max_error": transport[
                        "row_marginal_max_error"
                    ],
                    "column_marginal_max_error": transport[
                        "column_marginal_max_error"
                    ],
                    "transport_cost": transport["transport_cost"],
                }
            )
        except Exception as error:  # Recorded rather than silently discarded.
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
        record["runtime_seconds"] = float(time.perf_counter() - started)
        records.append(record)
    return records


def plot_results(
    pool_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    positions = np.arange(len(POOL_ORDER))
    for scenario in DEFAULT_SCENARIOS:
        subset = pool_summary.loc[pool_summary["scenario"] == scenario].set_index(
            "pool_variant"
        )
        if subset.empty:
            continue
        for axis, metric in zip(
            axes[:2],
            ("to_primary_oracle_rank_mae", "vs_default_jaccard_50"),
        ):
            means = np.asarray(
                [subset.loc[name, f"{metric}_mean"] for name in POOL_ORDER]
            )
            lower = np.asarray(
                [subset.loc[name, f"{metric}_ci_lower"] for name in POOL_ORDER]
            )
            upper = np.asarray(
                [subset.loc[name, f"{metric}_ci_upper"] for name in POOL_ORDER]
            )
            errors = np.vstack([means - lower, upper - means])
            errors[~np.isfinite(errors)] = 0.0
            axis.errorbar(
                positions,
                means,
                yerr=errors,
                marker="o",
                capsize=2.5,
                linewidth=1.7,
                color=COLORS[scenario],
                label=SCENARIO_LABELS[scenario],
            )
    axes[0].set_ylabel("Rank MAE to primary oracle")
    axes[1].set_ylabel("50% region Jaccard with default pool")
    for axis in axes[:2]:
        axis.set_xticks(positions, [POOL_LABELS[name] for name in POOL_ORDER], rotation=18)
        axis.grid(alpha=0.22)

    level_positions = np.arange(len(LEVELS))
    for scenario in DEFAULT_SCENARIOS:
        subset = bootstrap_summary.loc[
            bootstrap_summary["scenario"] == scenario
        ].set_index("level")
        if subset.empty:
            continue
        means = np.asarray(
            [subset.loc[level, "jaccard_mean"] for level in LEVELS]
        )
        lower = np.asarray(
            [subset.loc[level, "jaccard_ci_lower"] for level in LEVELS]
        )
        upper = np.asarray(
            [subset.loc[level, "jaccard_ci_upper"] for level in LEVELS]
        )
        errors = np.vstack([means - lower, upper - means])
        errors[~np.isfinite(errors)] = 0.0
        axes[2].errorbar(
            level_positions,
            means,
            yerr=errors,
            marker="o",
            capsize=2.5,
            linewidth=1.7,
            color=COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    axes[2].set_xticks(level_positions, [f"{int(100 * level)}%" for level in LEVELS])
    axes[2].set_xlabel("Quantile-region content")
    axes[2].set_ylabel("Bayesian-bootstrap region Jaccard")
    axes[2].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output_dir / "anchor_pool_bootstrap_stability.png", dpi=220)
    figure.savefig(output_dir / "anchor_pool_bootstrap_stability.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=20)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--density-ratio", type=float, default=20.0)
    parser.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS)
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.0)
    parser.add_argument("--mass-tolerance", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "anchor_pool_bootstrap_stability",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.sample_size = min(args.sample_size, 60)
        args.repeats = 1
        args.bootstrap_replicates = 3
        args.scenarios = ["sphere"]
    if args.sample_size < 20 or args.repeats < 1 or args.bootstrap_replicates < 1:
        raise ValueError("Invalid sample-size, repeats, or bootstrap-replicates.")
    if not np.isfinite(args.dirichlet_alpha) or args.dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet-alpha must be finite and positive.")
    unknown = sorted(set(args.scenarios) - set(GENERATORS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.scenarios) * args.repeats
    pool_records: List[Dict[str, Any]] = []
    bootstrap_records: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    for scenario_index, scenario in enumerate(args.scenarios):
        for repeat in range(args.repeats):
            dataset_started = time.perf_counter()
            random_state = int(
                args.seed + 100_003 * scenario_index + 1_009 * repeat
            )
            prepared = prepare_dataset(
                scenario,
                args.sample_size,
                args.density_ratio,
                random_state,
                args.winsor_quantile,
            )
            current_pool, base_ranks, pool_diagnostics = pool_audit(
                prepared,
                repeat,
                random_state,
                args.mass_tolerance,
            )
            current_bootstrap = bootstrap_audit(
                prepared,
                base_ranks,
                repeat,
                random_state,
                args.bootstrap_replicates,
                args.dirichlet_alpha,
                args.mass_tolerance,
            )
            pool_records.extend(current_pool)
            bootstrap_records.extend(current_bootstrap)
            dataset_runtime = float(time.perf_counter() - dataset_started)
            errors = sum(item["status"] != "ok" for item in current_bootstrap)
            diagnostics.append(
                {
                    "scenario": scenario,
                    "repeat": int(repeat),
                    "random_state": int(random_state),
                    "estimated_dimension": prepared["estimated_dimension"],
                    "anchor_index": prepared["anchor_index"],
                    "anchor_projection_error": prepared[
                        "anchor_projection_error"
                    ],
                    "graph": prepared["graph_diagnostics"],
                    "volume": prepared["volume_diagnostics"],
                    "estimated_transport": prepared["estimated_transport"],
                    "oracle_transport": prepared["oracle_transport"],
                    "pool": pool_diagnostics,
                    "bootstrap_errors": int(errors),
                    "runtime_seconds": dataset_runtime,
                }
            )
            completed += 1
            print(
                f"[{completed:>3}/{total}] {scenario}, repeat={repeat}, "
                f"bootstrap_errors={errors}, runtime={dataset_runtime:.2f}s",
                flush=True,
            )

    pool_frame = pd.DataFrame(pool_records)
    bootstrap_frame = pd.DataFrame(bootstrap_records)
    successful = bootstrap_frame.loc[bootstrap_frame["status"] == "ok"].copy()
    pool_metrics = [
        "to_primary_oracle_rank_spearman",
        "to_primary_oracle_rank_mae",
        "to_primary_oracle_jaccard_25",
        "to_primary_oracle_jaccard_50",
        "to_primary_oracle_jaccard_80",
        "to_primary_oracle_jaccard_90",
        "to_matched_oracle_rank_spearman",
        "to_matched_oracle_rank_mae",
        "vs_default_rank_spearman",
        "vs_default_rank_mae",
        "vs_default_jaccard_25",
        "vs_default_jaccard_50",
        "vs_default_jaccard_80",
        "vs_default_jaccard_90",
        "pool_size",
        "pool_radius",
    ]
    pool_summary = summarize(
        pool_frame,
        ["scenario", "scenario_label", "pool_variant", "pool_label"],
        pool_metrics,
    )

    bootstrap_metrics = [
        "vs_base_rank_spearman",
        "vs_base_rank_mae",
        "vs_base_jaccard_25",
        "vs_base_jaccard_50",
        "vs_base_jaccard_80",
        "vs_base_jaccard_90",
        "source_effective_sample_size",
        "pool_size",
        "pool_radius",
        "runtime_seconds",
    ]
    bootstrap_dataset = (
        successful.groupby(
            ["scenario", "scenario_label", "repeat", "random_state"],
            as_index=False,
        )[bootstrap_metrics]
        .mean()
    )
    bootstrap_dataset_summary = summarize(
        bootstrap_dataset,
        ["scenario", "scenario_label"],
        bootstrap_metrics,
    )
    long_records: List[Dict[str, Any]] = []
    for _, row in bootstrap_dataset.iterrows():
        for level in LEVELS:
            label = int(round(100 * level))
            long_records.append(
                {
                    "scenario": row["scenario"],
                    "scenario_label": row["scenario_label"],
                    "repeat": int(row["repeat"]),
                    "level": float(level),
                    "jaccard": float(row[f"vs_base_jaccard_{label}"]),
                }
            )
    bootstrap_long = pd.DataFrame(long_records)
    bootstrap_summary = summarize(
        bootstrap_long,
        ["scenario", "scenario_label", "level"],
        ["jaccard"],
    )

    pool_frame.to_csv(output_dir / "anchor_pool_records.csv", index=False)
    pool_summary.to_csv(output_dir / "anchor_pool_summary.csv", index=False)
    bootstrap_frame.to_csv(output_dir / "bootstrap_records.csv", index=False)
    bootstrap_dataset.to_csv(
        output_dir / "bootstrap_dataset_means.csv", index=False
    )
    bootstrap_dataset_summary.to_csv(
        output_dir / "bootstrap_dataset_summary.csv", index=False
    )
    bootstrap_summary.to_csv(output_dir / "bootstrap_region_summary.csv", index=False)
    plot_results(pool_summary, bootstrap_summary, output_dir)

    error_count = int(np.sum(bootstrap_frame["status"] != "ok"))
    result_summary = {
        "experiment": "current anchor-indexed AMQR pool and Bayesian-bootstrap stability",
        "pool_audit": (
            "Same estimated transport; local anchor-pool mass varied. Comparisons "
            "use both the primary default oracle and pool-matched oracle."
        ),
        "bootstrap_scope": (
            "Fixed-support Bayesian bootstrap of source weights; graph, target "
            "volume weights, scientific anchor, and n^{-1/2} pool mass are fixed."
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "bootstrap_attempts": int(len(bootstrap_frame)),
        "bootstrap_errors": error_count,
        "pool_summary": pool_summary.to_dict(orient="records"),
        "bootstrap_dataset_summary": bootstrap_dataset_summary.to_dict(
            orient="records"
        ),
        "bootstrap_region_summary": bootstrap_summary.to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result_summary), handle, ensure_ascii=False, indent=2)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "command": [str(item) for item in sys.argv],
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "parameters": vars(args),
        "pool_variants": pool_specifications(args.sample_size),
        "random_seed_scheme": "seed + 100003 * scenario_index + 1009 * repeat; bootstrap RNG adds 7919",
        "inferential_limit": (
            "Bayesian bootstrap is conditional on the observed support and excludes "
            "graph and intrinsic-volume estimation uncertainty."
        ),
        "diagnostics": diagnostics,
        "output_files": sorted(path.name for path in output_dir.iterdir()),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
