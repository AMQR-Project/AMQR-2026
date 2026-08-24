"""Entropic-regularization sensitivity for the current anchor-indexed AMQR.

This script deliberately uses the current anchor-free estimator rather than
the retired hard-anchor implementation.  For each simulated data set it fits
one transport from the empirical law to the estimated intrinsic-volume law,
then scores conditional transport rows against the same shrinking local
anchor pool.  Entropic regularization is scaled by the median nonzero squared
graph-geodesic distance so that epsilon ratios are comparable across supports.

The experiment reports both finite-support oracle recovery and deviation from
the unregularized current estimator.  Positive fixed epsilon changes the
transport target; it is therefore treated as a finite-sample sensitivity
analysis, not as evidence for the asymptotic theory of the exact estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
    region_mask,
    weighted_cdf_levels,
)


DEFAULT_SCENARIOS = ("sphere", "flat_torus", "curved_strip")
DEFAULT_EPSILON_RATIOS = (0.0, 0.003, 0.01, 0.03, 0.1)
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
    grouper: Any = list(group_columns)
    for key, group in frame.groupby(grouper, sort=False, dropna=False):
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


def fit_rank_field(
    distances: np.ndarray,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    anchor_index: int,
    epsilon: float,
    *,
    score_mass_tolerance: float,
    sinkhorn_num_iter_max: int,
    sinkhorn_stop_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    plan, transport = intrinsic_uniformization_plan(
        distances,
        source_weights,
        target_weights,
        entropic_regularization=float(epsilon),
        sinkhorn_method="sinkhorn_log",
        sinkhorn_num_iter_max=int(sinkhorn_num_iter_max),
        sinkhorn_stop_threshold=float(sinkhorn_stop_threshold),
    )
    scores, score = pooled_conditional_transport_wasserstein_scores(
        distances,
        plan,
        source_weights,
        anchor_index,
        mass_tolerance=float(score_mass_tolerance),
    )
    ranks = weighted_cdf_levels(scores, source_weights)
    return ranks, {
        "runtime_seconds": float(time.perf_counter() - started),
        "transport": transport,
        "score": score,
    }


def evaluate_ranks(
    ranks: np.ndarray, oracle_ranks: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    record = {
        f"{prefix}rank_spearman": finite_spearman(ranks, oracle_ranks),
        f"{prefix}rank_mae": float(np.mean(np.abs(ranks - oracle_ranks))),
    }
    for level in LEVELS:
        label = int(round(100 * level))
        record[f"{prefix}jaccard_{label}"] = jaccard(
            region_mask(ranks, level), region_mask(oracle_ranks, level)
        )
    return record


def run_dataset(
    scenario: str,
    n_samples: int,
    density_ratio: float,
    repeat: int,
    random_state: int,
    epsilon_ratios: Sequence[float],
    winsor_quantile: float,
    score_mass_tolerance: float,
    sinkhorn_num_iter_max: int,
    sinkhorn_stop_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
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
    source = np.full(n_samples, 1.0 / n_samples)
    oracle_target = 1.0 / np.maximum(sample.relative_density, 1e-12)
    oracle_target /= np.sum(oracle_target)
    anchor_index, anchor_projection_error = reference_anchor_index(sample)

    oracle_ranks, oracle_diagnostics = fit_rank_field(
        sample.true_distances,
        source,
        oracle_target,
        anchor_index,
        0.0,
        score_mass_tolerance=score_mass_tolerance,
        sinkhorn_num_iter_max=sinkhorn_num_iter_max,
        sinkhorn_stop_threshold=sinkhorn_stop_threshold,
    )
    upper = graph[np.triu_indices(n_samples, k=1)]
    positive = upper[upper > 0.0]
    epsilon_scale = float(np.median(positive ** 2))

    pending: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
    exact_ranks: np.ndarray | None = None
    for ratio in epsilon_ratios:
        epsilon = float(ratio) * epsilon_scale
        ranks, diagnostics = fit_rank_field(
            graph,
            source,
            target,
            anchor_index,
            epsilon,
            score_mass_tolerance=score_mass_tolerance,
            sinkhorn_num_iter_max=sinkhorn_num_iter_max,
            sinkhorn_stop_threshold=sinkhorn_stop_threshold,
        )
        if float(ratio) == 0.0:
            exact_ranks = ranks.copy()
        pending.append((float(ratio), ranks, diagnostics))
    if exact_ranks is None:
        raise ValueError("epsilon_ratios must contain zero for the exact reference.")

    records: List[Dict[str, Any]] = []
    for ratio, ranks, diagnostics in pending:
        transport = diagnostics["transport"]
        score = diagnostics["score"]
        record: Dict[str, Any] = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "regime": sample.regime,
            "n_samples": int(n_samples),
            "repeat": int(repeat),
            "random_state": int(random_state),
            "anchor_index": int(anchor_index),
            "anchor_projection_error": float(anchor_projection_error),
            "estimated_dimension": float(dimension),
            "epsilon_ratio": float(ratio),
            "epsilon": float(ratio * epsilon_scale),
            "epsilon_scale": epsilon_scale,
            "runtime_seconds": float(diagnostics["runtime_seconds"]),
            "solver": transport["solver"],
            "solver_converged": bool(transport["solver_converged"]),
            "solver_iterations": transport["solver_iterations"],
            "solver_final_error": transport["solver_final_error"],
            "row_marginal_max_error": transport["row_marginal_max_error"],
            "column_marginal_max_error": transport[
                "column_marginal_max_error"
            ],
            "transport_cost": transport["transport_cost"],
            "transport_plan_entropy": transport["transport_plan_entropy"],
            "mean_conditional_effective_targets": transport[
                "mean_conditional_effective_targets"
            ],
            "transport_plan_positive_fraction": transport[
                "transport_plan_positive_fraction"
            ],
            "anchor_pool_size": score["anchor_pool_size"],
            "anchor_pool_realized_mass": score[
                "anchor_pool_realized_source_mass"
            ],
            "anchor_pool_radius": score["anchor_pool_radius"],
            "anchor_conditional_support_size": score[
                "anchor_conditional_support_size"
            ],
            "mean_conditional_support_size": score[
                "mean_conditional_support_size"
            ],
        }
        record.update(evaluate_ranks(ranks, oracle_ranks))
        record.update(evaluate_ranks(ranks, exact_ranks, prefix="vs_exact_"))
        records.append(record)

    diagnostics = {
        "scenario": scenario,
        "repeat": int(repeat),
        "random_state": int(random_state),
        "estimated_dimension": float(dimension),
        "graph": graph_diagnostics,
        "volume": volume_diagnostics,
        "oracle": oracle_diagnostics,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return records, diagnostics


def paired_differences(records: pd.DataFrame) -> pd.DataFrame:
    key = ["scenario", "regime", "n_samples", "repeat", "random_state"]
    metrics = ["rank_spearman", "rank_mae"] + [
        f"jaccard_{int(round(100 * level))}" for level in LEVELS
    ]
    exact = records.loc[records["epsilon_ratio"] == 0.0, key + metrics].copy()
    exact = exact.rename(columns={metric: f"exact_{metric}" for metric in metrics})
    merged = records.merge(exact, on=key, validate="many_to_one")
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"exact_{metric}"]
    return merged


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("rank_mae", "Rank MAE to oracle", False),
        ("jaccard_50", "50% region Jaccard to oracle", False),
        ("vs_exact_rank_spearman", "Rank correlation with exact AMQR", True),
    )
    ratios = sorted(summary["epsilon_ratio"].unique())
    positions = np.arange(len(ratios))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    for axis, (metric, ylabel, unity) in zip(axes, metrics):
        for scenario in DEFAULT_SCENARIOS:
            subset = summary.loc[summary["scenario"] == scenario].set_index(
                "epsilon_ratio"
            )
            if subset.empty:
                continue
            means = np.asarray(
                [subset.loc[ratio, f"{metric}_mean"] for ratio in ratios]
            )
            lower = np.asarray(
                [subset.loc[ratio, f"{metric}_ci_lower"] for ratio in ratios]
            )
            upper = np.asarray(
                [subset.loc[ratio, f"{metric}_ci_upper"] for ratio in ratios]
            )
            errors = np.vstack([means - lower, upper - means])
            errors[~np.isfinite(errors)] = 0.0
            axis.errorbar(
                positions,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.7,
                capsize=2.5,
                color=COLORS[scenario],
                label=SCENARIO_LABELS[scenario],
            )
        if unity:
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        axis.set_xticks(positions, [f"{ratio:g}" for ratio in ratios])
        axis.set_xlabel(r"Scaled regularization $\varepsilon/\mathrm{median}(d_G^2)$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output_dir / "entropy_sensitivity.png", dpi=220)
    figure.savefig(output_dir / "entropy_sensitivity.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--density-ratio", type=float, default=20.0)
    parser.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS)
    )
    parser.add_argument(
        "--epsilon-ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_EPSILON_RATIOS),
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.0)
    parser.add_argument("--score-mass-tolerance", type=float, default=1e-8)
    parser.add_argument("--sinkhorn-num-iter-max", type=int, default=20_000)
    parser.add_argument("--sinkhorn-stop-threshold", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "anchor_indexed_entropy_sensitivity",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.sample_size = min(args.sample_size, 60)
        args.repeats = 1
        args.scenarios = ["sphere"]
        args.epsilon_ratios = [0.0, 0.003, 0.03]
    if args.sample_size < 20 or args.repeats < 1:
        raise ValueError("sample-size must be at least 20 and repeats must be positive.")
    unknown = sorted(set(args.scenarios) - set(GENERATORS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    ratios = sorted(set(float(value) for value in args.epsilon_ratios))
    if not ratios or ratios[0] < 0.0 or 0.0 not in ratios:
        raise ValueError("epsilon-ratios must be nonnegative and include zero.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.scenarios) * args.repeats
    all_records: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    for scenario_index, scenario in enumerate(args.scenarios):
        for repeat in range(args.repeats):
            random_state = int(
                args.seed + 100_003 * scenario_index + 1_009 * repeat
            )
            records, diagnostic = run_dataset(
                scenario=scenario,
                n_samples=args.sample_size,
                density_ratio=args.density_ratio,
                repeat=repeat,
                random_state=random_state,
                epsilon_ratios=ratios,
                winsor_quantile=args.winsor_quantile,
                score_mass_tolerance=args.score_mass_tolerance,
                sinkhorn_num_iter_max=args.sinkhorn_num_iter_max,
                sinkhorn_stop_threshold=args.sinkhorn_stop_threshold,
            )
            all_records.extend(records)
            diagnostics.append(diagnostic)
            completed += 1
            print(
                f"[{completed:>3}/{total}] {scenario}, repeat={repeat}, "
                f"runtime={diagnostic['runtime_seconds']:.2f}s",
                flush=True,
            )

    records = pd.DataFrame(all_records)
    metric_columns = [
        "rank_spearman",
        "rank_mae",
        "jaccard_25",
        "jaccard_50",
        "jaccard_80",
        "jaccard_90",
        "vs_exact_rank_spearman",
        "vs_exact_rank_mae",
        "vs_exact_jaccard_25",
        "vs_exact_jaccard_50",
        "vs_exact_jaccard_80",
        "vs_exact_jaccard_90",
        "runtime_seconds",
        "mean_conditional_effective_targets",
        "row_marginal_max_error",
        "column_marginal_max_error",
    ]
    summary = summarize(
        records,
        ["scenario", "scenario_label", "epsilon_ratio"],
        metric_columns,
    )
    paired = paired_differences(records)
    paired_metrics = [
        "delta_rank_spearman",
        "delta_rank_mae",
        "delta_jaccard_25",
        "delta_jaccard_50",
        "delta_jaccard_80",
        "delta_jaccard_90",
    ]
    paired_summary = summarize(
        paired,
        ["scenario", "scenario_label", "epsilon_ratio"],
        paired_metrics,
    )

    records.to_csv(output_dir / "entropy_records.csv", index=False)
    summary.to_csv(output_dir / "entropy_summary.csv", index=False)
    paired.to_csv(output_dir / "entropy_paired_records.csv", index=False)
    paired_summary.to_csv(output_dir / "entropy_paired_summary.csv", index=False)
    plot_results(summary, output_dir)

    converged = records.loc[records["epsilon_ratio"] > 0.0, "solver_converged"]
    result_summary = {
        "experiment": "current anchor-indexed AMQR entropic sensitivity",
        "estimator": "anchor-free intrinsic uniformization with pooled conditional-row W2 scores",
        "scope": "finite-sample sensitivity; fixed positive epsilon is not the primary asymptotic estimator",
        "elapsed_seconds": float(time.perf_counter() - started),
        "positive_epsilon_solver_convergence_rate": (
            float(np.mean(converged.astype(float))) if len(converged) else None
        ),
        "records": int(len(records)),
        "datasets": int(total),
        "summary": summary.to_dict(orient="records"),
        "paired_summary": paired_summary.to_dict(orient="records"),
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
        "epsilon_ratios_used": ratios,
        "random_seed_scheme": "seed + 100003 * scenario_index + 1009 * repeat",
        "score_mass_tolerance_note": (
            "Applied to conditional-row supports for every epsilon, including zero; "
            "rows are renormalized after thresholding."
        ),
        "diagnostics": diagnostics,
        "output_files": sorted(path.name for path in output_dir.iterdir()),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(manifest), handle, ensure_ascii=False, indent=2)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
