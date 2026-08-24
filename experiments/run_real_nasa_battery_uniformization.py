"""Pack-level OOS validation of AMQR on NASA battery aging curves.

Each observation is one constant-current reference-discharge cycle.  The
geometry is learned from voltage and temperature-rise curves resampled over
normalized discharged capacity.  Discharge duration/current-integrated
capacity is deliberately excluded from the fitted features and is used only
as an external state-of-health outcome.

The primary analysis uses regular and recommissioned first-life packs.  Model
assessment is out of sample at the battery-pack level.  Second-life packs are
scored only after fitting on all primary packs and are reported as an external
distribution-shift audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.anchored_uniformization import (  # noqa: E402
    AnchorIndexedIntrinsicUniformizer,
    extend_anchored_ranks,
    weighted_frechet_medoid,
)


DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "battery_alt_dataset"
)
GROUP_DIRECTORIES = {
    "regular": "regular_alt_batteries",
    "recommissioned": "recommissioned_batteries",
    "second_life": "second_life_batteries",
}
PRIMARY_GROUPS = ("regular", "recommissioned")
METHODS = ("amqr", "graph_radial", "ambient_radial")
METHOD_LABELS = {
    "amqr": "AMQR",
    "graph_radial": "Graph radial",
    "ambient_radial": "Ambient radial",
}
METHOD_COLORS = {
    "amqr": "#D1495B",
    "graph_radial": "#3B82A0",
    "ambient_radial": "#6B7280",
}
GROUP_COLORS = {
    "regular": "#2C7FB8",
    "recommissioned": "#E67E22",
    "second_life": "#7A5195",
}
REFERENCE_COLUMNS = [
    "start_time",
    "time",
    "mode",
    "voltage_load",
    "current_load",
    "temperature_battery",
    "mission_type",
]


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > 128:
            summary: Dict[str, Any] = {
                "stored_as": "numeric array summary",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if np.issubdtype(value.dtype, np.number):
                finite = np.asarray(value, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    summary.update(
                        {
                            "minimum": float(np.min(finite)),
                            "maximum": float(np.max(finite)),
                            "mean": float(np.mean(finite)),
                        }
                    )
            return summary
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if np.sum(valid) < 3:
        return float("nan")
    value = spearmanr(left_array[valid], right_array[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def weighted_cdf_at(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(reference_values, dtype=float).reshape(-1)
    weights = np.asarray(reference_weights, dtype=float).reshape(-1)
    queries = np.asarray(query_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    right = np.searchsorted(sorted_values, queries, side="right") - 1
    output = np.zeros_like(queries, dtype=float)
    valid = right >= 0
    output[valid] = cumulative[right[valid]]
    return np.clip(output, 0.0, 1.0)


def longest_contiguous_segment(
    frame: pd.DataFrame,
    maximum_gap_seconds: float = 10.0,
) -> pd.DataFrame:
    ordered = frame.sort_values("time", kind="mergesort").drop_duplicates("time")
    if len(ordered) < 2:
        return ordered
    gaps = np.diff(ordered["time"].to_numpy(float))
    split_points = np.flatnonzero(gaps > float(maximum_gap_seconds)) + 1
    pieces = np.split(np.arange(len(ordered)), split_points)
    largest = max(pieces, key=len)
    return ordered.iloc[largest]


def reference_cycle_record(
    frame: pd.DataFrame,
    grid: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray] | None:
    clean = frame[
        np.isfinite(frame["time"])
        & np.isfinite(frame["voltage_load"])
        & np.isfinite(frame["current_load"])
        & np.isfinite(frame["temperature_battery"])
        & frame["voltage_load"].between(4.0, 9.5)
        & frame["current_load"].between(1.5, 3.5)
        & frame["temperature_battery"].between(-10.0, 120.0)
    ]
    clean = longest_contiguous_segment(clean)
    if len(clean) < 500:
        return None

    elapsed = clean["time"].to_numpy(float)
    voltage = clean["voltage_load"].to_numpy(float)
    current = clean["current_load"].to_numpy(float)
    temperature = clean["temperature_battery"].to_numpy(float)
    elapsed -= elapsed[0]
    duration = float(elapsed[-1])
    if not np.isfinite(duration) or duration < 600.0:
        return None

    increments = 0.5 * (current[1:] + current[:-1]) * np.diff(elapsed) / 3600.0
    if np.any(increments < 0.0) or not np.isfinite(increments).all():
        return None
    discharged = np.concatenate(([0.0], np.cumsum(increments)))
    capacity = float(discharged[-1])
    if not 0.25 <= capacity <= 4.0:
        return None
    progress = discharged / capacity

    voltage_curve = np.interp(grid, progress, voltage)
    temperature_curve = np.interp(grid, progress, temperature)
    temperature_curve -= temperature_curve[0]
    record = {
        "n_reference_rows": int(len(clean)),
        "duration_seconds": duration,
        "capacity_ah": capacity,
        "mean_reference_current_a": float(np.mean(current)),
        "start_voltage_v": float(voltage_curve[0]),
        "end_voltage_v": float(voltage_curve[-1]),
        "temperature_rise_c": float(temperature_curve[-1]),
    }
    return record, voltage_curve.astype(np.float32), temperature_curve.astype(np.float32)


def extract_pack(
    path: Path,
    group: str,
    grid: np.ndarray,
    *,
    chunk_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    filtered_chunks = []
    total_rows = 0
    reference_rows = 0
    for chunk in pd.read_csv(
        path,
        usecols=REFERENCE_COLUMNS,
        chunksize=int(chunk_size),
        low_memory=False,
    ):
        # A small number of files contain textual logger artefacts in numeric
        # columns.  Coercion keeps the raw file auditable while treating only
        # the affected measurements as missing.
        for column in REFERENCE_COLUMNS[1:]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        total_rows += len(chunk)
        selected = chunk[
            chunk["mode"].eq(-1.0) & chunk["mission_type"].eq(0.0)
        ]
        reference_rows += len(selected)
        if len(selected):
            filtered_chunks.append(selected)
    if not filtered_chunks:
        raise ValueError(f"No reference-discharge rows found in {path}.")

    reference = pd.concat(filtered_chunks, ignore_index=True)
    metadata = []
    voltage_curves = []
    temperature_curves = []
    rejected = 0
    for start_time, frame in reference.groupby("start_time", sort=False):
        output = reference_cycle_record(frame, grid)
        if output is None:
            rejected += 1
            continue
        record, voltage_curve, temperature_curve = output
        record.update(
            {
                "pack_id": path.stem,
                "battery_group": group,
                "start_time": str(start_time),
                "_curve_position": len(voltage_curves),
            }
        )
        metadata.append(record)
        voltage_curves.append(voltage_curve)
        temperature_curves.append(temperature_curve)
    if not metadata:
        raise ValueError(f"All reference cycles were rejected in {path}.")

    frame = pd.DataFrame(metadata)
    frame["start_datetime"] = pd.to_datetime(
        frame["start_time"], errors="coerce", format="mixed"
    )
    frame = frame.sort_values("start_datetime", kind="mergesort")
    order = frame["_curve_position"].to_numpy(int)
    frame = frame.drop(columns="_curve_position").reset_index(drop=True)
    voltage_array = np.asarray(voltage_curves, dtype=np.float32)[order]
    temperature_array = np.asarray(temperature_curves, dtype=np.float32)[order]
    frame["cycle_index"] = np.arange(len(frame), dtype=int)
    frame["life_fraction"] = (
        frame["cycle_index"] / max(1, len(frame) - 1)
    )
    initial_count = min(3, len(frame))
    initial_capacity = float(np.median(frame.loc[: initial_count - 1, "capacity_ah"]))
    frame["initial_capacity_ah"] = initial_capacity
    frame["capacity_retention"] = frame["capacity_ah"] / initial_capacity
    frame["capacity_loss"] = 1.0 - frame["capacity_retention"]
    frame["cycle_uid"] = [f"{path.stem}:{value}" for value in frame["cycle_index"]]
    diagnostics = {
        "file": str(path),
        "group": group,
        "total_rows": int(total_rows),
        "reference_rows": int(reference_rows),
        "accepted_reference_cycles": int(len(frame)),
        "rejected_reference_cycles": int(rejected),
        "initial_capacity_ah": initial_capacity,
        "final_capacity_retention": float(frame["capacity_retention"].iloc[-1]),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return frame, voltage_array, temperature_array, diagnostics


def extract_reference_cycles(
    dataset_dir: Path,
    *,
    grid_size: int,
    chunk_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, int(grid_size))
    metadata_blocks = []
    voltage_blocks = []
    temperature_blocks = []
    diagnostics: Dict[str, Any] = {"packs": {}}
    offset = 0
    for group, directory in GROUP_DIRECTORIES.items():
        paths = sorted((dataset_dir / directory).glob("battery*.csv"))
        if not paths:
            raise FileNotFoundError(f"No battery CSV files found in {dataset_dir / directory}.")
        for path in paths:
            print(f"Extracting {group}/{path.name} ...", flush=True)
            frame, voltage, temperature, pack_diagnostics = extract_pack(
                path,
                group,
                grid,
                chunk_size=chunk_size,
            )
            frame["array_index"] = np.arange(offset, offset + len(frame), dtype=int)
            offset += len(frame)
            metadata_blocks.append(frame)
            voltage_blocks.append(voltage)
            temperature_blocks.append(temperature)
            diagnostics["packs"][path.stem] = pack_diagnostics
            print(
                f"  accepted={len(frame)}, final retention="
                f"{pack_diagnostics['final_capacity_retention']:.3f}",
                flush=True,
            )
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    voltage = np.concatenate(voltage_blocks, axis=0)
    temperature = np.concatenate(temperature_blocks, axis=0)
    diagnostics.update(
        {
            "n_packs": int(metadata["pack_id"].nunique()),
            "n_cycles": int(len(metadata)),
            "grid_size": int(grid_size),
            "groups": metadata.groupby("battery_group")["pack_id"].nunique().to_dict(),
        }
    )
    return metadata, voltage, temperature, diagnostics


def load_or_extract(
    dataset_dir: Path,
    processed_dir: Path,
    *,
    grid_size: int,
    chunk_size: int,
    rebuild: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nasa_battery_alt_reference_cycles_grid{int(grid_size)}"
    metadata_path = processed_dir / f"{stem}.csv"
    arrays_path = processed_dir / f"{stem}.npz"
    diagnostics_path = processed_dir / f"{stem}_diagnostics.json"
    if not rebuild and metadata_path.exists() and arrays_path.exists():
        print(f"Loading extracted reference-cycle cache: {arrays_path}", flush=True)
        metadata = pd.read_csv(metadata_path)
        arrays = np.load(arrays_path)
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.exists()
            else {"cache": "loaded without extraction diagnostics"}
        )
        return metadata, arrays["voltage"], arrays["temperature"], diagnostics

    metadata, voltage, temperature, diagnostics = extract_reference_cycles(
        dataset_dir,
        grid_size=grid_size,
        chunk_size=chunk_size,
    )
    metadata.to_csv(metadata_path, index=False)
    np.savez_compressed(arrays_path, voltage=voltage, temperature=temperature)
    diagnostics_path.write_text(
        json.dumps(json_ready(diagnostics), indent=2), encoding="utf-8"
    )
    return metadata, voltage, temperature, diagnostics


def systematic_pack_sample(
    metadata: pd.DataFrame,
    maximum_per_pack: int,
) -> np.ndarray:
    selected = []
    for _, frame in metadata.groupby("pack_id", sort=True):
        indices = frame.index.to_numpy(int)
        maximum = min(int(maximum_per_pack), len(indices))
        if len(indices) <= maximum:
            selected.extend(indices.tolist())
            continue
        mandatory = indices[: min(3, len(indices))].tolist() + [int(indices[-1])]
        grid = np.linspace(0, len(indices) - 1, maximum, dtype=int)
        choices = list(dict.fromkeys([*mandatory, *indices[grid].tolist()]))
        if len(choices) > maximum:
            protected = set(mandatory)
            removable = [value for value in choices if value not in protected]
            keep_removable = maximum - len(protected)
            positions = np.linspace(0, len(removable) - 1, keep_removable, dtype=int)
            choices = mandatory + [removable[position] for position in positions]
        selected.extend(sorted(set(choices))[:maximum])
    return np.asarray(sorted(selected), dtype=int)


def standardized_functional_features(
    voltage: np.ndarray,
    temperature: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    blocks = []
    diagnostics: Dict[str, Any] = {}
    for name, values in (("voltage", voltage), ("temperature_rise", temperature)):
        values = np.asarray(values, dtype=float)
        location = np.mean(values[train_indices], axis=0)
        scale = np.std(values[train_indices], axis=0, ddof=1)
        positive = scale[scale > 1e-10]
        fallback = float(np.median(positive)) if positive.size else 1.0
        floor = max(1e-8, fallback * 1e-3)
        constant = scale <= floor
        scale[constant] = 1.0
        standardized = (values - location) / scale
        standardized /= np.sqrt(values.shape[1])
        blocks.append(standardized)
        diagnostics[name] = {
            "grid_size": int(values.shape[1]),
            "near_constant_grid_points": int(np.sum(constant)),
            "scale_floor": floor,
        }
    return np.concatenate(blocks, axis=1).astype(np.float32), diagnostics


def pack_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("pack_id")["pack_id"].transform("size").to_numpy(float)
    n_packs = metadata["pack_id"].nunique()
    weights = 1.0 / (float(n_packs) * counts)
    return weights / np.sum(weights)


def stratified_pack_folds(
    metadata: pd.DataFrame,
    n_folds: int,
    random_state: int,
) -> Dict[str, int]:
    rng = np.random.default_rng(random_state)
    assignments: Dict[str, int] = {}
    for group, frame in metadata.groupby("battery_group", sort=True):
        packs = np.asarray(sorted(frame["pack_id"].unique()), dtype=object)
        rng.shuffle(packs)
        for position, pack in enumerate(packs):
            assignments[str(pack)] = int(position % int(n_folds))
    return assignments


def select_early_anchor(
    intrinsic_distances: np.ndarray,
    metadata: pd.DataFrame,
    source_weights: np.ndarray,
    early_fraction: float,
) -> Tuple[int, Dict[str, Any]]:
    early = np.flatnonzero(
        (metadata["life_fraction"].to_numpy(float) <= float(early_fraction))
        | (metadata["cycle_index"].to_numpy(int) <= 2)
    )
    if len(early) < 3:
        raise ValueError("Too few early reference cycles to define an anchor.")
    early_weights = source_weights[early]
    early_weights /= np.sum(early_weights)
    local_index, objective = weighted_frechet_medoid(
        intrinsic_distances[np.ix_(early, early)], early_weights
    )
    anchor = int(early[local_index])
    return anchor, {
        "candidate_count": int(len(early)),
        "candidate_source_mass": float(np.sum(source_weights[early])),
        "selection": "weighted intrinsic medoid among beginning-of-life supported cycles",
        "objective": float(objective[local_index]),
        "pack_id": str(metadata.iloc[anchor]["pack_id"]),
        "cycle_index": int(metadata.iloc[anchor]["cycle_index"]),
        "life_fraction": float(metadata.iloc[anchor]["life_fraction"]),
        "capacity_retention_used_only_for_reporting": float(
            metadata.iloc[anchor]["capacity_retention"]
        ),
    }


def fit_and_extend(
    features: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    k_graph: int,
    k_volume: int,
    k_dimension: int,
    winsor_quantile: float,
    early_fraction: float,
    interpolation_neighbors: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    started = time.perf_counter()
    train_features = np.asarray(features[train_indices], dtype=float)
    test_features = np.asarray(features[test_indices], dtype=float)
    train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
    source = pack_balanced_weights(train_metadata)
    local = cdist(train_features, train_features)
    model = AnchorIndexedIntrinsicUniformizer(
        k_graph=min(int(k_graph), len(train_indices) - 1),
        k_volume=min(int(k_volume), len(train_indices) - 1),
        k_dimension=min(int(k_dimension), len(train_indices) - 1),
        winsor_quantile=float(winsor_quantile),
        entropic_regularization=0.0,
        ensure_connected=True,
    ).fit(local, source_weights=source)
    anchor, anchor_diagnostics = select_early_anchor(
        model.intrinsic_distances_,
        train_metadata,
        source,
        early_fraction,
    )
    train_amqr_ranks, train_amqr_scores, rank_diagnostics = model.ranks_for_anchor(
        anchor, return_diagnostics=True
    )

    cross = cdist(test_features, train_features)
    amqr_ranks, amqr_scores, graph_cross, extension_diagnostics = extend_anchored_ranks(
        cross,
        model.intrinsic_distances_,
        train_amqr_scores,
        source,
        graph_distance_scale=float(
            model.graph_diagnostics_["median_graph_distance_before_scaling"]
        ),
        graph_connections=min(int(k_graph), len(train_indices)),
        interpolation_neighbors=min(int(interpolation_neighbors), len(train_indices)),
    )

    train_graph_scores = model.intrinsic_distances_[:, anchor]
    test_graph_scores = graph_cross[:, anchor]
    graph_ranks = weighted_cdf_at(train_graph_scores, source, test_graph_scores)
    train_ambient_scores = local[:, anchor]
    test_ambient_scores = cross[:, anchor]
    ambient_ranks = weighted_cdf_at(train_ambient_scores, source, test_ambient_scores)
    outputs = {
        "amqr": {"rank": amqr_ranks, "score": amqr_scores},
        "graph_radial": {"rank": graph_ranks, "score": test_graph_scores},
        "ambient_radial": {"rank": ambient_ranks, "score": test_ambient_scores},
    }
    diagnostics = {
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_train_packs": int(train_metadata["pack_id"].nunique()),
        "anchor": anchor_diagnostics,
        "uniformizer": model.diagnostics(),
        "anchor_scoring": rank_diagnostics,
        "extension": extension_diagnostics,
        "runtime_seconds": float(time.perf_counter() - started),
        "train_amqr_rank_range": [
            float(np.min(train_amqr_ranks)),
            float(np.max(train_amqr_ranks)),
        ],
    }
    return outputs, diagnostics


def run_pack_oos(
    metadata: pd.DataFrame,
    voltage: np.ndarray,
    temperature: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
    k_graph: int,
    k_volume: int,
    k_dimension: int,
    winsor_quantile: float,
    early_fraction: float,
    interpolation_neighbors: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    primary = metadata["battery_group"].isin(PRIMARY_GROUPS).to_numpy(bool)
    primary_indices = np.flatnonzero(primary)
    primary_metadata = metadata.iloc[primary_indices].reset_index(drop=True)
    assignments = stratified_pack_folds(primary_metadata, n_folds, random_state)
    records = []
    fold_diagnostics: Dict[str, Any] = {}

    for fold in range(int(n_folds)):
        held_out_packs = sorted(
            pack for pack, assigned in assignments.items() if assigned == fold
        )
        test_mask = primary_metadata["pack_id"].isin(held_out_packs).to_numpy(bool)
        train_local = np.flatnonzero(~test_mask)
        test_local = np.flatnonzero(test_mask)
        features, scaling_diagnostics = standardized_functional_features(
            voltage[primary_indices], temperature[primary_indices], train_local
        )
        print(
            f"Fold {fold + 1}/{n_folds}: train packs="
            f"{primary_metadata.iloc[train_local]['pack_id'].nunique()}, "
            f"test packs={held_out_packs}",
            flush=True,
        )
        outputs, diagnostics = fit_and_extend(
            features,
            primary_metadata,
            train_local,
            test_local,
            k_graph=k_graph,
            k_volume=k_volume,
            k_dimension=k_dimension,
            winsor_quantile=winsor_quantile,
            early_fraction=early_fraction,
            interpolation_neighbors=interpolation_neighbors,
        )
        diagnostics["held_out_packs"] = held_out_packs
        diagnostics["feature_scaling"] = scaling_diagnostics
        fold_diagnostics[str(fold)] = diagnostics
        base = primary_metadata.iloc[test_local].copy()
        base["evaluation_set"] = "primary_pack_oos"
        base["fold"] = fold
        for method in METHODS:
            block = base.copy()
            block["method"] = method
            block["rank"] = outputs[method]["rank"]
            block["raw_score"] = outputs[method]["score"]
            records.append(block)

    oos = pd.concat(records, ignore_index=True)

    # External distribution-shift audit: fit all first-life packs and score the
    # three second-life packs without incorporating them into the geometry.
    second_indices = np.flatnonzero(~primary)
    final_features, scaling_diagnostics = standardized_functional_features(
        voltage, temperature, primary_indices
    )
    if len(second_indices):
        print("Scoring second-life packs from the primary-pack fit ...", flush=True)
        outputs, final_diagnostics = fit_and_extend(
            final_features,
            metadata,
            primary_indices,
            second_indices,
            k_graph=k_graph,
            k_volume=k_volume,
            k_dimension=k_dimension,
            winsor_quantile=winsor_quantile,
            early_fraction=early_fraction,
            interpolation_neighbors=interpolation_neighbors,
        )
        final_diagnostics["feature_scaling"] = scaling_diagnostics
        final_diagnostics["purpose"] = "external second-life distribution-shift audit"
        fold_diagnostics["second_life_external"] = final_diagnostics
        base = metadata.iloc[second_indices].copy()
        base["evaluation_set"] = "external_second_life"
        base["fold"] = -1
        external_records = []
        for method in METHODS:
            block = base.copy()
            block["method"] = method
            block["rank"] = outputs[method]["rank"]
            block["raw_score"] = outputs[method]["score"]
            external_records.append(block)
        oos = pd.concat([oos, *external_records], ignore_index=True)
    return oos, fold_diagnostics, final_features


def bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_bootstrap: int = 2000,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    samples = rng.choice(array, size=(int(n_bootstrap), len(array)), replace=True)
    estimates = np.mean(samples, axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_scores(
    scores: pd.DataFrame,
    *,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    pack_records = []
    summary_records = []
    for evaluation_set, set_frame in scores.groupby("evaluation_set", sort=False):
        for method, frame in set_frame.groupby("method", sort=False):
            pack_values = []
            for pack_id, pack in frame.groupby("pack_id", sort=True):
                record = {
                    "evaluation_set": evaluation_set,
                    "method": method,
                    "pack_id": pack_id,
                    "battery_group": str(pack["battery_group"].iloc[0]),
                    "n_cycles": int(len(pack)),
                    "spearman_capacity_loss": safe_spearman(
                        pack["rank"], pack["capacity_loss"]
                    ),
                    "spearman_life_fraction": safe_spearman(
                        pack["rank"], pack["life_fraction"]
                    ),
                }
                pack_records.append(record)
                pack_values.append(record)
            pack_frame = pd.DataFrame(pack_values)
            cap_values = pack_frame["spearman_capacity_loss"].to_numpy(float)
            life_values = pack_frame["spearman_life_fraction"].to_numpy(float)
            cap_ci = bootstrap_mean_ci(cap_values, rng)
            life_ci = bootstrap_mean_ci(life_values, rng)
            late = frame["life_fraction"].to_numpy(float) >= 0.8
            fade = frame["capacity_loss"].to_numpy(float) >= 0.15
            late_auc = (
                float(roc_auc_score(late, frame["rank"]))
                if np.unique(late).size == 2
                else float("nan")
            )
            fade_auc = (
                float(roc_auc_score(fade, frame["rank"]))
                if np.unique(fade).size == 2
                else float("nan")
            )
            outer = frame["rank"].to_numpy(float) > 0.8
            inner = frame["rank"].to_numpy(float) <= 0.5
            summary_records.append(
                {
                    "evaluation_set": evaluation_set,
                    "method": method,
                    "n_packs": int(frame["pack_id"].nunique()),
                    "n_cycles": int(len(frame)),
                    "overall_spearman_capacity_loss": safe_spearman(
                        frame["rank"], frame["capacity_loss"]
                    ),
                    "mean_pack_spearman_capacity_loss": float(np.nanmean(cap_values)),
                    "median_pack_spearman_capacity_loss": float(np.nanmedian(cap_values)),
                    "mean_pack_spearman_capacity_loss_ci_low": cap_ci[0],
                    "mean_pack_spearman_capacity_loss_ci_high": cap_ci[1],
                    "overall_spearman_life_fraction": safe_spearman(
                        frame["rank"], frame["life_fraction"]
                    ),
                    "mean_pack_spearman_life_fraction": float(np.nanmean(life_values)),
                    "mean_pack_spearman_life_fraction_ci_low": life_ci[0],
                    "mean_pack_spearman_life_fraction_ci_high": life_ci[1],
                    "late_life_auc": late_auc,
                    "capacity_fade_15pct_auc": fade_auc,
                    "outer_20_fraction": float(np.mean(outer)),
                    "outer_20_mean_capacity_loss": float(
                        np.mean(frame.loc[outer, "capacity_loss"]) if np.any(outer) else np.nan
                    ),
                    "inner_50_mean_capacity_loss": float(
                        np.mean(frame.loc[inner, "capacity_loss"]) if np.any(inner) else np.nan
                    ),
                    "outer_minus_inner_capacity_loss": float(
                        np.mean(frame.loc[outer, "capacity_loss"])
                        - np.mean(frame.loc[inner, "capacity_loss"])
                        if np.any(outer) and np.any(inner)
                        else np.nan
                    ),
                }
            )
    pack_metrics = pd.DataFrame(pack_records)
    summary = pd.DataFrame(summary_records)

    comparisons = []
    primary = pack_metrics[pack_metrics["evaluation_set"].eq("primary_pack_oos")]
    for baseline in METHODS[1:]:
        wide = primary.pivot(index="pack_id", columns="method", values="spearman_capacity_loss")
        paired = wide[["amqr", baseline]].dropna()
        differences = paired["amqr"] - paired[baseline]
        if len(differences) and np.any(np.abs(differences) > 1e-12):
            test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = float("nan")
            p_value = float("nan")
        comparisons.append(
            {
                "metric": "pack-level Spearman(rank, capacity loss)",
                "comparison": f"AMQR - {METHOD_LABELS[baseline]}",
                "n_packs": int(len(differences)),
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "wilcoxon_statistic_exploratory": statistic,
                "wilcoxon_p_value_exploratory": p_value,
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    if len(comparison_frame):
        comparison_frame["holm_threshold_order"] = np.arange(1, len(comparison_frame) + 1)
    return summary, pack_metrics, comparison_frame


def plot_validation_scatter(scores: pd.DataFrame, output_path: Path) -> None:
    frame = scores[scores["evaluation_set"].eq("primary_pack_oos")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        method_frame = frame[frame["method"].eq(method)]
        for group, group_frame in method_frame.groupby("battery_group"):
            axis.scatter(
                group_frame["capacity_loss"] * 100.0,
                group_frame["rank"],
                s=18,
                alpha=0.66,
                color=GROUP_COLORS[group],
                label=group.replace("_", " "),
                edgecolors="none",
            )
        axis.axhline(0.8, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Capacity loss from pack baseline (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Out-of-pack structural rank")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle("NASA battery aging: external capacity validation", y=1.14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rank_trajectories(scores: pd.DataFrame, output_path: Path) -> None:
    frame = scores[scores["evaluation_set"].eq("primary_pack_oos")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 6)
    for axis, method in zip(axes, METHODS):
        method_frame = frame[frame["method"].eq(method)]
        for pack_id, pack in method_frame.groupby("pack_id", sort=True):
            group = str(pack["battery_group"].iloc[0])
            ordered = pack.sort_values("life_fraction")
            axis.plot(
                ordered["life_fraction"],
                ordered["rank"],
                color=GROUP_COLORS[group],
                alpha=0.28,
                linewidth=0.9,
            )
        binned = method_frame.copy()
        binned["life_bin"] = pd.cut(
            binned["life_fraction"], bins=bins, include_lowest=True, labels=False
        )
        pack_bin = (
            binned.groupby(["pack_id", "life_bin"], observed=True)["rank"]
            .mean()
            .reset_index()
        )
        mean_curve = pack_bin.groupby("life_bin")["rank"].agg(["mean", "sem"])
        centers = 0.5 * (bins[:-1] + bins[1:])
        positions = mean_curve.index.to_numpy(int)
        axis.errorbar(
            centers[positions],
            mean_curve["mean"],
            yerr=mean_curve["sem"],
            color=METHOD_COLORS[method],
            linewidth=2.6,
            marker="o",
            markersize=4,
            capsize=2,
            label="pack-balanced mean",
        )
        axis.axhline(0.8, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Observed life fraction")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Out-of-pack structural rank")
    fig.suptitle("Pack-level rank trajectories", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[summary["evaluation_set"].eq("primary_pack_oos")].set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    metrics = [
        ("mean_pack_spearman_capacity_loss", "Mean within-pack Spearman"),
        ("late_life_auc", "Late-life AUROC"),
        ("outer_minus_inner_capacity_loss", "Outer-minus-inner capacity loss"),
    ]
    for axis, (metric, title) in zip(axes, metrics):
        values = [float(frame.loc[method, metric]) for method in METHODS]
        axis.bar(
            np.arange(len(METHODS)),
            values,
            color=[METHOD_COLORS[method] for method in METHODS],
            width=0.68,
        )
        axis.set_xticks(np.arange(len(METHODS)))
        axis.set_xticklabels([METHOD_LABELS[item] for item in METHODS], rotation=18)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        for position, value in enumerate(values):
            axis.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("NASA battery AMQR: pack-level OOS validation", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    output_path: Path,
    random_state: int,
) -> None:
    primary = metadata["battery_group"].isin(PRIMARY_GROUPS).to_numpy(bool)
    coordinates = PCA(n_components=2, random_state=random_state).fit_transform(features[primary])
    primary_meta = metadata.loc[primary, ["cycle_uid", "battery_group"]].copy()
    amqr = scores[
        scores["evaluation_set"].eq("primary_pack_oos") & scores["method"].eq("amqr")
    ][["cycle_uid", "rank"]]
    display = primary_meta.merge(amqr, on="cycle_uid", how="left", validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    scatter = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=display["rank"], cmap="viridis",
        vmin=0.0, vmax=1.0, s=25, alpha=0.82, edgecolors="none"
    )
    axes[0].set_title("OOF AMQR rank")
    fig.colorbar(scatter, ax=axes[0], label="rank")
    for group in PRIMARY_GROUPS:
        mask = display["battery_group"].eq(group).to_numpy(bool)
        axes[1].scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=25, alpha=0.68,
            color=GROUP_COLORS[group], label=group.replace("_", " "), edgecolors="none"
        )
    axes[1].set_title("Battery protocol group")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("PCA 1 (display only)")
        axis.set_ylabel("PCA 2 (display only)")
        axis.grid(alpha=0.18)
    fig.suptitle("Reference-discharge curve geometry", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_curves(
    voltage: np.ndarray,
    temperature: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    primary = metadata[metadata["battery_group"].isin(PRIMARY_GROUPS)]
    quantiles = [0.1, 0.5, 0.9]
    labels = ["low fade", "intermediate", "high fade"]
    targets = np.quantile(primary["capacity_loss"], quantiles)
    chosen = [
        int((primary["capacity_loss"] - target).abs().idxmin()) for target in targets
    ]
    grid = np.linspace(0.0, 1.0, voltage.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    colors = ["#2C7FB8", "#FDAE61", "#D7191C"]
    for index, label, color in zip(chosen, labels, colors):
        fade = 100.0 * float(metadata.loc[index, "capacity_loss"])
        axes[0].plot(grid, voltage[index], color=color, linewidth=2.0, label=f"{label} ({fade:.1f}%)")
        axes[1].plot(grid, temperature[index], color=color, linewidth=2.0)
    axes[0].set_ylabel("Pack voltage (V)")
    axes[1].set_ylabel("Temperature rise (C)")
    for axis in axes:
        axis.set_xlabel("Normalized discharged capacity")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[0].set_title("Voltage morphology")
    axes[1].set_title("Thermal morphology")
    fig.suptitle("Representative reference-discharge functions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pack-level OOS AMQR validation on NASA battery aging data."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "real_nasa_battery_uniformization",
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument(
        "--max-cycles-per-pack",
        type=int,
        default=100,
        help="Deterministic per-pack cap; the default retains all cycles in this dataset.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k-graph", type=int, default=18)
    parser.add_argument("--k-volume", type=int, default=18)
    parser.add_argument("--k-dimension", type=int, default=12)
    parser.add_argument("--winsor-quantile", type=float, default=0.02)
    parser.add_argument("--early-fraction", type=float, default=0.12)
    parser.add_argument("--interpolation-neighbors", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=20260815)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.max_cycles_per_pack = min(args.max_cycles_per_pack, 8)
        args.folds = min(args.folds, 3)

    metadata, voltage, temperature, extraction_diagnostics = load_or_extract(
        args.dataset_dir,
        args.processed_dir,
        grid_size=args.grid_size,
        chunk_size=args.chunk_size,
        rebuild=args.rebuild_cache,
    )
    selected = systematic_pack_sample(metadata, args.max_cycles_per_pack)
    analysis_metadata = metadata.iloc[selected].copy().reset_index(drop=True)
    analysis_metadata["source_array_index"] = selected
    analysis_voltage = voltage[selected]
    analysis_temperature = temperature[selected]
    print(
        f"Analysis sample: {len(analysis_metadata)} cycles from "
        f"{analysis_metadata['pack_id'].nunique()} packs.",
        flush=True,
    )

    scores, fit_diagnostics, display_features = run_pack_oos(
        analysis_metadata,
        analysis_voltage,
        analysis_temperature,
        n_folds=args.folds,
        random_state=args.random_state,
        k_graph=args.k_graph,
        k_volume=args.k_volume,
        k_dimension=args.k_dimension,
        winsor_quantile=args.winsor_quantile,
        early_fraction=args.early_fraction,
        interpolation_neighbors=args.interpolation_neighbors,
    )
    summary, pack_metrics, comparisons = evaluate_scores(
        scores, random_state=args.random_state
    )

    analysis_metadata.to_csv(args.output_dir / "battery_analysis_cycles.csv", index=False)
    scores.to_csv(args.output_dir / "battery_oos_scores.csv", index=False)
    summary.to_csv(args.output_dir / "battery_metric_summary.csv", index=False)
    pack_metrics.to_csv(args.output_dir / "battery_pack_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "battery_method_comparisons.csv", index=False)
    plot_validation_scatter(scores, args.output_dir / "battery_capacity_validation.png")
    plot_rank_trajectories(scores, args.output_dir / "battery_rank_trajectories.png")
    plot_metric_summary(summary, args.output_dir / "battery_metric_summary.png")
    plot_embedding(
        display_features,
        analysis_metadata,
        scores,
        args.output_dir / "battery_amqr_embedding.png",
        args.random_state,
    )
    plot_representative_curves(
        analysis_voltage,
        analysis_temperature,
        analysis_metadata,
        args.output_dir / "battery_representative_curves.png",
    )

    total_runtime = float(time.perf_counter() - started)
    report = {
        "analysis": "NASA battery pack-level OOS anchored intrinsic uniformization",
        "experiment_type": "real-data functional analysis",
        "verification_status": "completed, single deterministic run",
        "dataset_dir": str(args.dataset_dir),
        "primary_groups": list(PRIMARY_GROUPS),
        "external_group": "second_life",
        "feature_definition": (
            "voltage and temperature-rise functions over normalized discharged capacity; "
            "capacity and duration excluded from the fit"
        ),
        "anchor_definition": (
            "weighted intrinsic medoid among supported beginning-of-life cycles "
            f"(life fraction <= {args.early_fraction})"
        ),
        "validation_design": (
            f"{args.folds}-fold battery-pack OOS; pack-balanced source weights; "
            "capacity loss is an external outcome"
        ),
        "parameters": vars(args),
        "extraction": extraction_diagnostics,
        "fit": fit_diagnostics,
        "metric_summary": summary.to_dict(orient="records"),
        "method_comparisons": comparisons.to_dict(orient="records"),
        "runtime_seconds": total_runtime,
    }
    (args.output_dir / "battery_experiment_summary.json").write_text(
        json.dumps(json_ready(report), indent=2), encoding="utf-8"
    )
    print("\nPrimary pack-level OOS metric summary:", flush=True)
    print(
        summary[summary["evaluation_set"].eq("primary_pack_oos")].to_string(index=False),
        flush=True,
    )
    print(f"\nCompleted in {total_runtime:.1f} seconds: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
