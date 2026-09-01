"""Distance-only intrinsic uniformization and anchor-indexed ranks.

This module implements two related estimators used in the Adaptive Manifold
Quantile Regions (AMQR) manuscript.
:class:`AnchorIndexedIntrinsicUniformizer` is the population-aligned estimator:
it estimates one anchor-free transport from the empirical
law to an intrinsic-volume reference, then evaluates any user-specified anchor
without refitting that transport.  Because a discrete optimal plan need not be
Monge, its primary structural score is the target-space Wasserstein-2 distance
between an observation's conditional transport row and a source-weighted local
pool of rows around the anchor.  The shrinking pool avoids treating integrated
plan convergence as convergence of one data-selected row.  In the Monge limit
this reduces to distance from the transported observation to the transported
anchor.

The older :class:`AnchoredIntrinsicUniformizer` is retained as a comparator and
for backward compatibility.  It locates a graph medoid and hard-constrains that
row separately in the transport problem.

Both estimators can build intrinsic graph distances, estimate intrinsic
dimension and normalized volume weights, and calibrate scores into empirical
ranks.  No coordinates, analytic geodesics, or manifold volume form are
required once pairwise local dissimilarities have been supplied.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import ot
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph


def _validate_distance_matrix(distances: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(distances, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if len(matrix) < 3:
        raise ValueError(f"{name} must contain at least three observations.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    if np.min(matrix) < -1e-10:
        raise ValueError(f"{name} contains negative distances.")
    matrix = np.maximum(matrix, 0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _validate_weights(
    weights: Optional[np.ndarray], n_samples: int, name: str
) -> np.ndarray:
    if weights is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != n_samples:
        raise ValueError(f"{name} must have length {n_samples}.")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return values / np.sum(values)


def estimate_intrinsic_dimension(
    local_distances: np.ndarray,
    k_neighbors: int = 15,
    minimum: float = 1.0,
    maximum: float = 10.0,
) -> float:
    """Estimate intrinsic dimension with a Levina--Bickel local MLE."""
    distances = _validate_distance_matrix(local_distances, "local_distances")
    n_samples = len(distances)
    k = min(max(4, int(k_neighbors)), n_samples - 1)
    masked = distances + np.eye(n_samples) * 1e100
    neighbours = np.sort(masked, axis=1)[:, :k]
    outer = neighbours[:, -1]
    log_ratios = np.log(
        (outer[:, None] + 1e-12) / (neighbours[:, :-1] + 1e-12)
    )
    inverse_dimension = float(np.mean(log_ratios))
    if not np.isfinite(inverse_dimension) or inverse_dimension <= 0.0:
        raise RuntimeError("The intrinsic-dimension estimate was not finite.")
    return float(np.clip(1.0 / inverse_dimension, minimum, maximum))


def graph_geodesic_distances(
    local_distances: np.ndarray,
    k_neighbors: int = 20,
    ensure_connected: bool = True,
    maximum_neighbors: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct symmetric k-NN graph shortest-path distances.

    When requested, the neighbourhood size is increased one point at a time
    until the graph is connected.  The effective value is returned so that
    this deterministic repair is visible in diagnostics.
    """
    distances = _validate_distance_matrix(local_distances, "local_distances")
    n_samples = len(distances)
    requested_k = min(max(2, int(k_neighbors)), n_samples - 1)
    max_k = (
        n_samples - 1
        if maximum_neighbors is None
        else min(max(requested_k, int(maximum_neighbors)), n_samples - 1)
    )

    graph = None
    n_components = n_samples
    effective_k = requested_k
    initial_components = None
    while effective_k <= max_k:
        candidate = kneighbors_graph(
            distances,
            n_neighbors=effective_k,
            mode="distance",
            metric="precomputed",
            include_self=False,
        )
        candidate = candidate.maximum(candidate.T)
        n_components = int(connected_components(candidate, directed=False)[0])
        if initial_components is None:
            initial_components = n_components
        graph = candidate
        if n_components == 1 or not ensure_connected:
            break
        effective_k += 1

    if graph is None or n_components != 1:
        raise RuntimeError(
            "The k-NN graph is disconnected. Increase maximum_neighbors or "
            "analyse connected supports separately."
        )

    geodesic = np.asarray(shortest_path(graph, directed=False), dtype=float)
    if not np.isfinite(geodesic).all():
        raise RuntimeError("Graph shortest paths contain non-finite values.")
    np.fill_diagonal(geodesic, 0.0)
    positive = geodesic[geodesic > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    geodesic /= scale

    diagnostics = {
        "requested_k_neighbors": requested_k,
        "effective_k_neighbors": int(effective_k),
        "initial_graph_components": int(initial_components),
        "graph_components": int(n_components),
        "median_graph_distance_before_scaling": scale,
        "undirected_edges": int(graph.nnz // 2),
    }
    return geodesic, diagnostics


def estimate_uniform_volume_weights(
    local_distances: np.ndarray,
    intrinsic_dimension: float,
    k_neighbors: int = 20,
    winsor_quantile: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate normalized intrinsic volume by inverse k-NN density."""
    distances = _validate_distance_matrix(local_distances, "local_distances")
    n_samples = len(distances)
    k = min(max(3, int(k_neighbors)), n_samples - 1)
    if intrinsic_dimension <= 0.0 or not np.isfinite(intrinsic_dimension):
        raise ValueError("intrinsic_dimension must be finite and positive.")
    if not 0.0 <= winsor_quantile < 0.5:
        raise ValueError("winsor_quantile must lie in [0, 0.5).")

    masked = distances + np.eye(n_samples) * 1e100
    radius = np.sort(masked, axis=1)[:, k - 1]
    raw = np.maximum(radius, 1e-12) ** float(intrinsic_dimension)
    lower = float(np.min(raw))
    upper = float(np.max(raw))
    if winsor_quantile > 0.0:
        lower, upper = np.quantile(
            raw, [winsor_quantile, 1.0 - winsor_quantile]
        )
        raw = np.clip(raw, lower, upper)
    weights = raw / np.sum(raw)
    diagnostics = {
        "k_neighbors": int(k),
        "intrinsic_dimension": float(intrinsic_dimension),
        "winsor_quantile": float(winsor_quantile),
        "winsor_lower": float(lower),
        "winsor_upper": float(upper),
        "minimum_weight": float(np.min(weights)),
        "maximum_weight": float(np.max(weights)),
        "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
    }
    return weights, diagnostics


def weighted_frechet_medoid(
    intrinsic_distances: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """Return the weighted intrinsic L1 medoid and its objective values."""
    distances = _validate_distance_matrix(
        intrinsic_distances, "intrinsic_distances"
    )
    weights = _validate_weights(source_weights, len(distances), "source_weights")
    objective = distances @ weights
    return int(np.argmin(objective)), objective


def _weighted_cdf_levels(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Right-continuous weighted empirical CDF evaluated at every value."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    right = np.searchsorted(sorted_values, values, side="right") - 1
    return np.clip(cumulative[right], 0.0, 1.0)


def _weighted_cdf_at(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    """Evaluate a weighted empirical CDF at arbitrary query values."""
    values = np.asarray(reference_values, dtype=float).reshape(-1)
    weights = np.asarray(reference_weights, dtype=float).reshape(-1)
    queries = np.asarray(query_values, dtype=float)
    if len(values) != len(weights):
        raise ValueError("reference_values and reference_weights must align.")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    right = np.searchsorted(sorted_values, queries, side="right") - 1
    output = np.zeros_like(queries, dtype=float)
    valid = right >= 0
    output[valid] = cumulative[right[valid]]
    return np.clip(output, 0.0, 1.0)


def extend_anchored_ranks(
    ambient_cross_distances: np.ndarray,
    train_intrinsic_distances: np.ndarray,
    train_raw_scores: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
    *,
    graph_distance_scale: float = 1.0,
    graph_connections: int = 20,
    interpolation_neighbors: int = 15,
    interpolation_bandwidth: Optional[float] = None,
    batch_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extend fitted anchored ranks to new responses using distances only.

    New observations are attached to the fitted training graph through their
    nearest ambient-distance neighbours.  The fitted raw structural score is
    then smoothed in graph distance with the *same source weights* used by the
    conditional fit and calibrated with the corresponding weighted empirical
    CDF.  Including the source weights in both operations is essential: an
    unweighted response-neighbour average is not controlled by a conditional
    source-weighted training error.

    Parameters
    ----------
    ambient_cross_distances:
        Pairwise local dissimilarities from test rows to training columns.
    train_intrinsic_distances:
        Fitted, scaled graph-geodesic distances among training observations.
    train_raw_scores:
        Raw structural scores returned by an anchored uniformization fit.
    source_weights:
        Fitted conditional source weights.  Uniform weights are used when
        omitted.
    graph_distance_scale:
        The pre-scaling median graph distance reported by
        :func:`graph_geodesic_distances`.  Cross edges are divided by this
        value before being combined with the scaled training graph.
    interpolation_neighbors:
        Effective number of source observations used to select the automatic
        response-kernel bandwidth.  For asymptotic use this value should grow
        while remaining negligible relative to the source effective sample
        size.
    interpolation_bandwidth:
        Optional fixed bandwidth in scaled graph-distance units.  If omitted,
        a source-weighted nearest-neighbour radius is estimated from the
        training graph.
    """
    started = time.perf_counter()
    intrinsic = _validate_distance_matrix(
        train_intrinsic_distances, "train_intrinsic_distances"
    )
    n_train = len(intrinsic)
    cross = np.asarray(ambient_cross_distances, dtype=float)
    if cross.ndim != 2 or cross.shape[1] != n_train:
        raise ValueError(
            "ambient_cross_distances must have one column per training sample."
        )
    if not np.isfinite(cross).all() or np.any(cross < 0.0):
        raise ValueError("ambient_cross_distances must be finite and nonnegative.")
    raw_scores = np.asarray(train_raw_scores, dtype=float).reshape(-1)
    if len(raw_scores) != n_train or not np.isfinite(raw_scores).all():
        raise ValueError("train_raw_scores must be finite and match the training set.")
    source = _validate_weights(source_weights, n_train, "source_weights")
    scale = float(graph_distance_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("graph_distance_scale must be finite and positive.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")

    n_test = cross.shape[0]
    connection_k = min(max(2, int(graph_connections)), n_train)
    source_effective_size = float(1.0 / np.sum(source ** 2))
    interpolation_k = min(
        max(3, int(interpolation_neighbors)),
        max(3, int(np.floor(source_effective_size))),
    )
    scaled_cross = cross / scale
    graph_cross = np.empty((n_test, n_train), dtype=float)
    for start in range(0, n_test, int(batch_size)):
        stop = min(start + int(batch_size), n_test)
        block = scaled_cross[start:stop]
        attachment = np.argpartition(
            block, kth=connection_k - 1, axis=1
        )[:, :connection_k]
        attachment_edges = np.take_along_axis(block, attachment, axis=1)
        candidate_paths = (
            attachment_edges[:, :, None] + intrinsic[attachment]
        )
        graph_cross[start:stop] = np.min(candidate_paths, axis=1)

    if interpolation_bandwidth is None:
        target_mass = min(
            0.5,
            max(
                interpolation_k / source_effective_size,
                1.01 * float(np.max(source)),
            ),
        )
        training_order = np.argsort(intrinsic, axis=1, kind="mergesort")
        ordered_source = source[training_order]
        cumulative_source = np.cumsum(ordered_source, axis=1)
        radius_positions = np.argmax(cumulative_source >= target_mass, axis=1)
        selected_neighbours = training_order[
            np.arange(n_train), radius_positions
        ]
        training_radii = intrinsic[np.arange(n_train), selected_neighbours]
        positive_radii = training_radii[training_radii > 1e-12]
        if positive_radii.size:
            radius_order = np.argsort(training_radii, kind="mergesort")
            radius_cumulative = np.cumsum(source[radius_order])
            median_position = int(np.searchsorted(radius_cumulative, 0.5, side="left"))
            bandwidth = float(training_radii[radius_order[median_position]])
        else:
            positive = intrinsic[intrinsic > 0.0]
            bandwidth = float(np.median(positive)) if positive.size else 1.0
        bandwidth_source = "source-weighted training k-NN radius"
    else:
        bandwidth = float(interpolation_bandwidth)
        target_mass = None
        bandwidth_source = "supplied"
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("interpolation_bandwidth must be finite and positive.")

    log_weights = -0.5 * (graph_cross / bandwidth) ** 2
    log_weights += np.log(source)[None, :]
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    interpolation_weights = np.exp(log_weights)
    interpolation_weights /= np.sum(interpolation_weights, axis=1, keepdims=True)
    test_raw_scores = interpolation_weights @ raw_scores
    test_ranks = _weighted_cdf_at(raw_scores, source, test_raw_scores)
    interpolation_ess = 1.0 / np.sum(interpolation_weights ** 2, axis=1)
    diagnostics = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "graph_connections": int(connection_k),
        "interpolation_neighbors": int(interpolation_k),
        "source_effective_sample_size": source_effective_size,
        "interpolation_target_source_mass": target_mass,
        "interpolation_bandwidth": bandwidth,
        "interpolation_bandwidth_source": bandwidth_source,
        "minimum_interpolation_effective_size": float(np.min(interpolation_ess))
        if n_test
        else None,
        "median_interpolation_effective_size": float(np.median(interpolation_ess))
        if n_test
        else None,
        "maximum_interpolation_effective_size": float(np.max(interpolation_ess))
        if n_test
        else None,
        "graph_distance_scale": scale,
        "minimum_test_rank": float(np.min(test_ranks)) if n_test else None,
        "maximum_test_rank": float(np.max(test_ranks)) if n_test else None,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return test_ranks, test_raw_scores, graph_cross, diagnostics


def anchored_transport_ranks(
    intrinsic_distances: np.ndarray,
    source_weights: Optional[np.ndarray],
    target_weights: np.ndarray,
    center_index: int,
    num_iter_max: int = 2_000_000,
    entropic_regularization: float = 0.0,
    sinkhorn_method: str = "sinkhorn_log",
    sinkhorn_num_iter_max: int = 20_000,
    sinkhorn_stop_threshold: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute hard-anchored same-support OT ranks.

    The entire source row at ``center_index`` is fixed to the target pole at
    the same support point.  If the target pole has insufficient mass, it is
    lifted to the source atom mass and all other target weights are rescaled.

    With ``entropic_regularization=0`` (the default), the residual transport
    is the original exact Earth-Mover solution.  Positive values solve the
    residual problem with Sinkhorn regularization while leaving the hard
    center anchor untouched.  The regularization has the same units as the
    squared intrinsic-distance cost.
    """
    started = time.perf_counter()
    distances = _validate_distance_matrix(
        intrinsic_distances, "intrinsic_distances"
    )
    n_samples = len(distances)
    source = _validate_weights(source_weights, n_samples, "source_weights")
    target = _validate_weights(target_weights, n_samples, "target_weights").copy()
    center_index = int(center_index)
    if not 0 <= center_index < n_samples:
        raise IndexError("center_index is outside the observed support.")
    regularization = float(entropic_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("entropic_regularization must be finite and nonnegative.")
    if int(sinkhorn_num_iter_max) < 1:
        raise ValueError("sinkhorn_num_iter_max must be positive.")
    if (
        not np.isfinite(sinkhorn_stop_threshold)
        or float(sinkhorn_stop_threshold) <= 0.0
    ):
        raise ValueError("sinkhorn_stop_threshold must be finite and positive.")

    anchor_mass = float(source[center_index])
    original_pole_mass = float(target[center_index])
    target_adjusted = False
    if target[center_index] < anchor_mass:
        remaining_scale = (1.0 - anchor_mass) / (1.0 - target[center_index])
        target *= remaining_scale
        target[center_index] = anchor_mass
        target /= np.sum(target)
        target_adjusted = True

    plan = np.zeros((n_samples, n_samples), dtype=float)
    plan[center_index, center_index] = anchor_mass
    remaining_rows = np.delete(np.arange(n_samples), center_index)
    source_remaining = source[remaining_rows]
    target_remaining = target.copy()
    target_remaining[center_index] -= anchor_mass
    target_remaining[np.abs(target_remaining) < 1e-15] = 0.0
    target_remaining = np.maximum(target_remaining, 0.0)
    target_remaining *= np.sum(source_remaining) / np.sum(target_remaining)

    reduced_cost = distances[remaining_rows] ** 2
    solver_log: Dict[str, Any] = {}
    if regularization == 0.0:
        reduced = ot.emd(
            source_remaining,
            target_remaining,
            reduced_cost,
            numItermax=int(num_iter_max),
        )
        plan[remaining_rows] = reduced
        solver_name = "emd"
        solver_iterations = None
        solver_final_error = None
        solver_converged = True
        active_target_columns = np.arange(n_samples)
    else:
        # Removing exactly zero-mass columns avoids log(0) in log-domain
        # Sinkhorn implementations.  The omitted entries remain zero in the
        # full transport plan.
        active_target_columns = np.flatnonzero(target_remaining > 1e-15)
        if active_target_columns.size == 0:
            raise RuntimeError("No positive residual target mass remains.")
        reduced, solver_log = ot.sinkhorn(
            source_remaining,
            target_remaining[active_target_columns],
            reduced_cost[:, active_target_columns],
            regularization,
            method=str(sinkhorn_method),
            numItermax=int(sinkhorn_num_iter_max),
            stopThr=float(sinkhorn_stop_threshold),
            log=True,
            warn=False,
        )
        reduced = np.asarray(reduced, dtype=float)
        if not np.isfinite(reduced).all() or np.any(reduced < -1e-12):
            raise RuntimeError(
                "The entropically regularized transport solver returned an "
                "invalid plan. Increase entropic_regularization."
            )
        reduced = np.maximum(reduced, 0.0)
        plan[np.ix_(remaining_rows, active_target_columns)] = reduced
        solver_name = str(sinkhorn_method)
        solver_iterations = solver_log.get("niter")
        errors = np.asarray(solver_log.get("err", []), dtype=float).reshape(-1)
        solver_final_error = float(errors[-1]) if errors.size else None
        solver_converged = bool(
            solver_final_error is not None
            and np.isfinite(solver_final_error)
            and solver_final_error <= float(sinkhorn_stop_threshold)
        )

    target_radii = distances[center_index]
    target_levels = _weighted_cdf_levels(target_radii, target)
    conditional_plan = plan / source[:, None]
    raw_scores = conditional_plan @ target_levels
    ranks = _weighted_cdf_levels(raw_scores, source)

    positive_plan = plan[plan > 0.0]
    conditional_entropy = -np.sum(
        np.where(
            conditional_plan > 0.0,
            conditional_plan * np.log(np.maximum(conditional_plan, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    diagnostics = {
        "center_index": center_index,
        "anchor_mass": anchor_mass,
        "original_target_pole_mass": original_pole_mass,
        "adjusted_target_pole_mass": float(target[center_index]),
        "target_mass_adjusted": bool(target_adjusted),
        "transport_cost": float(np.sum(plan * distances ** 2)),
        "entropic_regularization": regularization,
        "solver": solver_name,
        "solver_iterations": (
            None if solver_iterations is None else int(solver_iterations)
        ),
        "solver_final_error": solver_final_error,
        "solver_converged": bool(solver_converged),
        "sinkhorn_method": str(sinkhorn_method) if regularization > 0.0 else None,
        "sinkhorn_num_iter_max": (
            int(sinkhorn_num_iter_max) if regularization > 0.0 else None
        ),
        "sinkhorn_stop_threshold": (
            float(sinkhorn_stop_threshold) if regularization > 0.0 else None
        ),
        "active_target_columns": int(len(active_target_columns)),
        "transport_plan_entropy": float(
            -np.sum(positive_plan * np.log(positive_plan))
        ),
        "mean_conditional_entropy": float(np.sum(source * conditional_entropy)),
        "mean_conditional_effective_targets": float(
            np.sum(source * np.exp(conditional_entropy))
        ),
        "transport_plan_positive_fraction": float(np.mean(plan > 1e-12)),
        "row_marginal_max_error": float(
            np.max(np.abs(plan.sum(axis=1) - source))
        ),
        "column_marginal_max_error": float(
            np.max(np.abs(plan.sum(axis=0) - target))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return ranks, raw_scores, plan, target, diagnostics


def intrinsic_uniformization_plan(
    intrinsic_distances: np.ndarray,
    source_weights: Optional[np.ndarray],
    target_weights: np.ndarray,
    num_iter_max: int = 2_000_000,
    entropic_regularization: float = 0.0,
    sinkhorn_method: str = "sinkhorn_log",
    sinkhorn_num_iter_max: int = 20_000,
    sinkhorn_stop_threshold: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate one anchor-free transport to intrinsic uniform measure.

    Rows are source observations and columns are the same observed support
    equipped with estimated intrinsic-volume weights.  The conditional row
    ``plan[i] / source_weights[i]`` is a measure-valued empirical analogue of
    ``T_P(Y_i)``.  In the Monge case it is a point mass.
    """
    started = time.perf_counter()
    distances = _validate_distance_matrix(
        intrinsic_distances, "intrinsic_distances"
    )
    n_samples = len(distances)
    source = _validate_weights(source_weights, n_samples, "source_weights")
    target = _validate_weights(target_weights, n_samples, "target_weights")
    regularization = float(entropic_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("entropic_regularization must be finite and nonnegative.")
    if int(sinkhorn_num_iter_max) < 1:
        raise ValueError("sinkhorn_num_iter_max must be positive.")
    if (
        not np.isfinite(sinkhorn_stop_threshold)
        or float(sinkhorn_stop_threshold) <= 0.0
    ):
        raise ValueError("sinkhorn_stop_threshold must be finite and positive.")

    cost = distances ** 2
    solver_log: Dict[str, Any] = {}
    if regularization == 0.0:
        plan = ot.emd(source, target, cost, numItermax=int(num_iter_max))
        solver_name = "emd"
        solver_iterations = None
        solver_final_error = None
        solver_converged = True
    else:
        plan, solver_log = ot.sinkhorn(
            source,
            target,
            cost,
            regularization,
            method=str(sinkhorn_method),
            numItermax=int(sinkhorn_num_iter_max),
            stopThr=float(sinkhorn_stop_threshold),
            log=True,
            warn=False,
        )
        solver_name = str(sinkhorn_method)
        solver_iterations = solver_log.get("niter")
        errors = np.asarray(solver_log.get("err", []), dtype=float).reshape(-1)
        solver_final_error = float(errors[-1]) if errors.size else None
        solver_converged = bool(
            solver_final_error is not None
            and np.isfinite(solver_final_error)
            and solver_final_error <= float(sinkhorn_stop_threshold)
        )

    plan = np.asarray(plan, dtype=float)
    if plan.shape != (n_samples, n_samples):
        raise RuntimeError("The uniformization solver returned a plan of wrong shape.")
    if not np.isfinite(plan).all() or np.any(plan < -1e-12):
        raise RuntimeError("The uniformization solver returned an invalid plan.")
    plan = np.maximum(plan, 0.0)
    conditional = plan / source[:, None]
    conditional_entropy = -np.sum(
        np.where(
            conditional > 0.0,
            conditional * np.log(np.maximum(conditional, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    positive = plan[plan > 0.0]
    diagnostics = {
        "method": "anchor-free intrinsic uniformization",
        "transport_cost": float(np.sum(plan * cost)),
        "entropic_regularization": regularization,
        "solver": solver_name,
        "solver_iterations": (
            None if solver_iterations is None else int(solver_iterations)
        ),
        "solver_final_error": solver_final_error,
        "solver_converged": bool(solver_converged),
        "transport_plan_entropy": float(-np.sum(positive * np.log(positive))),
        "mean_conditional_entropy": float(np.sum(source * conditional_entropy)),
        "mean_conditional_effective_targets": float(
            np.sum(source * np.exp(conditional_entropy))
        ),
        "transport_plan_positive_fraction": float(np.mean(plan > 1e-12)),
        "row_marginal_max_error": float(
            np.max(np.abs(plan.sum(axis=1) - source))
        ),
        "column_marginal_max_error": float(
            np.max(np.abs(plan.sum(axis=0) - target))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return plan, diagnostics


def conditional_transport_wasserstein_scores(
    intrinsic_distances: np.ndarray,
    transport_plan: np.ndarray,
    source_weights: Optional[np.ndarray],
    anchor_index: int,
    *,
    mass_tolerance: float = 1e-14,
    num_iter_max: int = 200_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Distance from every conditional transport row to one anchor row.

    The score is the target-space Wasserstein-2 distance

    ``W2(plan[i] / source[i], plan[a] / source[a])``.

    It depends only on the fitted anchor-free plan and target distance matrix.
    If every conditional row is a Dirac mass, this is exactly the target-space
    distance between empirical images of observations ``i`` and ``a``.
    """
    started = time.perf_counter()
    distances = _validate_distance_matrix(
        intrinsic_distances, "intrinsic_distances"
    )
    n_samples = len(distances)
    source = _validate_weights(source_weights, n_samples, "source_weights")
    plan = np.asarray(transport_plan, dtype=float)
    if plan.shape != (n_samples, n_samples):
        raise ValueError("transport_plan must align with intrinsic_distances.")
    if not np.isfinite(plan).all() or np.any(plan < -1e-12):
        raise ValueError("transport_plan must be finite and nonnegative.")
    plan = np.maximum(plan, 0.0)
    row_error = float(np.max(np.abs(plan.sum(axis=1) - source)))
    if row_error > 1e-7:
        raise ValueError("transport_plan row marginals do not match source_weights.")
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < n_samples:
        raise IndexError("anchor_index is outside the observed support.")
    tolerance = float(mass_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mass_tolerance must be finite and nonnegative.")

    conditionals = plan / source[:, None]
    anchor = conditionals[anchor_index]
    anchor_support = np.flatnonzero(anchor > tolerance)
    if anchor_support.size == 0:
        raise RuntimeError("The anchor conditional transport row has no mass.")
    anchor_mass = anchor[anchor_support]
    anchor_mass /= np.sum(anchor_mass)
    cost = distances ** 2
    scores_squared = np.empty(n_samples, dtype=float)
    support_sizes = np.empty(n_samples, dtype=int)
    for index in range(n_samples):
        row = conditionals[index]
        row_support = np.flatnonzero(row > tolerance)
        if row_support.size == 0:
            raise RuntimeError(f"Conditional transport row {index} has no mass.")
        row_mass = row[row_support]
        row_mass /= np.sum(row_mass)
        support_sizes[index] = int(row_support.size)
        scores_squared[index] = float(
            ot.emd2(
                row_mass,
                anchor_mass,
                cost[np.ix_(row_support, anchor_support)],
                numItermax=int(num_iter_max),
            )
        )
    scores_squared = np.maximum(scores_squared, 0.0)
    scores = np.sqrt(scores_squared)
    scores[anchor_index] = 0.0
    diagnostics = {
        "anchor_index": anchor_index,
        "score_metric": "target-space Wasserstein-2 between conditional rows",
        "anchor_conditional_support_size": int(anchor_support.size),
        "minimum_conditional_support_size": int(np.min(support_sizes)),
        "maximum_conditional_support_size": int(np.max(support_sizes)),
        "mean_conditional_support_size": float(np.mean(support_sizes)),
        "minimum_score": float(np.min(scores)),
        "maximum_score": float(np.max(scores)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return scores, diagnostics


def pooled_conditional_transport_wasserstein_scores(
    intrinsic_distances: np.ndarray,
    transport_plan: np.ndarray,
    source_weights: Optional[np.ndarray],
    anchor_index: int,
    *,
    anchor_pool_mass: Optional[float] = None,
    minimum_anchor_rows: int = 3,
    mass_tolerance: float = 1e-14,
    num_iter_max: int = 200_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Score conditional transport rows against a locally pooled anchor.

    Rows nearest to the supported anchor in intrinsic distance are accumulated
    until they contain ``anchor_pool_mass`` of the fitted source law.  Their
    conditional transport measures are then averaged with source weights.  If
    no mass is supplied, the default is ``n_eff**(-1/2)``; hence the pool mass
    vanishes while its effective number of observations grows.
    """
    started = time.perf_counter()
    distances = _validate_distance_matrix(
        intrinsic_distances, "intrinsic_distances"
    )
    n_samples = len(distances)
    source = _validate_weights(source_weights, n_samples, "source_weights")
    plan = np.asarray(transport_plan, dtype=float)
    if plan.shape != (n_samples, n_samples):
        raise ValueError("transport_plan must align with intrinsic_distances.")
    if not np.isfinite(plan).all() or np.any(plan < -1e-12):
        raise ValueError("transport_plan must be finite and nonnegative.")
    plan = np.maximum(plan, 0.0)
    row_error = float(np.max(np.abs(plan.sum(axis=1) - source)))
    if row_error > 1e-7:
        raise ValueError("transport_plan row marginals do not match source_weights.")
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < n_samples:
        raise IndexError("anchor_index is outside the observed support.")
    tolerance = float(mass_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("mass_tolerance must be finite and nonnegative.")
    minimum_rows = min(max(1, int(minimum_anchor_rows)), n_samples)

    source_effective_size = float(1.0 / np.sum(source ** 2))
    if anchor_pool_mass is None:
        requested_mass = float(source_effective_size ** -0.5)
        mass_source = "automatic n_eff^(-1/2)"
    else:
        requested_mass = float(anchor_pool_mass)
        mass_source = "supplied"
    if not np.isfinite(requested_mass) or not 0.0 < requested_mass <= 1.0:
        raise ValueError("anchor_pool_mass must lie in (0, 1].")

    anchor_order = np.argsort(distances[anchor_index], kind="mergesort")
    cumulative_mass = np.cumsum(source[anchor_order])
    crossing = int(np.searchsorted(cumulative_mass, requested_mass, side="left")) + 1
    pool_count = min(n_samples, max(minimum_rows, crossing))
    pool_indices = anchor_order[:pool_count]
    realized_mass = float(np.sum(source[pool_indices]))
    pooled_anchor = np.sum(plan[pool_indices], axis=0) / realized_mass
    pooled_anchor = np.maximum(pooled_anchor, 0.0)
    pooled_anchor /= np.sum(pooled_anchor)
    anchor_support = np.flatnonzero(pooled_anchor > tolerance)
    if anchor_support.size == 0:
        raise RuntimeError("The pooled anchor transport kernel has no mass.")
    anchor_mass = pooled_anchor[anchor_support]
    anchor_mass /= np.sum(anchor_mass)

    conditionals = plan / source[:, None]
    cost = distances ** 2
    scores_squared = np.empty(n_samples, dtype=float)
    support_sizes = np.empty(n_samples, dtype=int)
    for index in range(n_samples):
        row = conditionals[index]
        row_support = np.flatnonzero(row > tolerance)
        if row_support.size == 0:
            raise RuntimeError(f"Conditional transport row {index} has no mass.")
        row_mass = row[row_support]
        row_mass /= np.sum(row_mass)
        support_sizes[index] = int(row_support.size)
        scores_squared[index] = float(
            ot.emd2(
                row_mass,
                anchor_mass,
                cost[np.ix_(row_support, anchor_support)],
                numItermax=int(num_iter_max),
            )
        )
    scores = np.sqrt(np.maximum(scores_squared, 0.0))
    diagnostics = {
        "anchor_index": anchor_index,
        "score_metric": "target-space Wasserstein-2 to a pooled anchor kernel",
        "anchor_pool_mass_source": mass_source,
        "anchor_pool_requested_source_mass": requested_mass,
        "anchor_pool_realized_source_mass": realized_mass,
        "anchor_pool_size": int(pool_count),
        "anchor_pool_radius": float(distances[anchor_index, pool_indices[-1]]),
        "anchor_pool_indices": pool_indices.copy(),
        "anchor_conditional_support_size": int(anchor_support.size),
        "source_effective_sample_size": source_effective_size,
        "minimum_conditional_support_size": int(np.min(support_sizes)),
        "maximum_conditional_support_size": int(np.max(support_sizes)),
        "mean_conditional_support_size": float(np.mean(support_sizes)),
        "minimum_score": float(np.min(scores)),
        "maximum_score": float(np.max(scores)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return scores, diagnostics


class AnchorIndexedIntrinsicUniformizer:
    """One anchor-free fit followed by arbitrary anchor-indexed ranks.

    ``fit`` estimates geometry, intrinsic volume, and one transport plan.  The
    fitted object can then be queried repeatedly with ``ranks_for_anchor``
    without changing the transport.  This estimator is designed to correspond
    to a population uniformization ``T_P`` that is independent of the selected
    anchor.
    """

    def __init__(
        self,
        *,
        k_graph: int = 20,
        k_volume: int = 20,
        k_dimension: int = 15,
        intrinsic_dimension: Optional[float] = None,
        winsor_quantile: float = 0.0,
        anchor_pool_mass: Optional[float] = None,
        minimum_anchor_rows: int = 3,
        ensure_connected: bool = True,
        maximum_graph_neighbors: Optional[int] = None,
        entropic_regularization: float = 0.0,
        sinkhorn_method: str = "sinkhorn_log",
        sinkhorn_num_iter_max: int = 20_000,
        sinkhorn_stop_threshold: float = 1e-9,
    ) -> None:
        self.k_graph = int(k_graph)
        self.k_volume = int(k_volume)
        self.k_dimension = int(k_dimension)
        self.intrinsic_dimension = intrinsic_dimension
        self.winsor_quantile = float(winsor_quantile)
        self.anchor_pool_mass = anchor_pool_mass
        self.minimum_anchor_rows = int(minimum_anchor_rows)
        self.ensure_connected = bool(ensure_connected)
        self.maximum_graph_neighbors = maximum_graph_neighbors
        self.entropic_regularization = float(entropic_regularization)
        self.sinkhorn_method = str(sinkhorn_method)
        self.sinkhorn_num_iter_max = int(sinkhorn_num_iter_max)
        self.sinkhorn_stop_threshold = float(sinkhorn_stop_threshold)
        self.is_fitted_ = False

    def fit(
        self,
        local_distances: np.ndarray,
        *,
        intrinsic_distances: Optional[np.ndarray] = None,
        source_weights: Optional[np.ndarray] = None,
    ) -> "AnchorIndexedIntrinsicUniformizer":
        started = time.perf_counter()
        local = _validate_distance_matrix(local_distances, "local_distances")
        n_samples = len(local)
        source = _validate_weights(source_weights, n_samples, "source_weights")
        if intrinsic_distances is None:
            intrinsic, graph_diagnostics = graph_geodesic_distances(
                local,
                k_neighbors=self.k_graph,
                ensure_connected=self.ensure_connected,
                maximum_neighbors=self.maximum_graph_neighbors,
            )
        else:
            intrinsic = _validate_distance_matrix(
                intrinsic_distances, "intrinsic_distances"
            )
            graph_diagnostics = {
                "source": "supplied intrinsic distance matrix",
                "requested_k_neighbors": None,
                "effective_k_neighbors": None,
            }
        dimension = (
            estimate_intrinsic_dimension(local, self.k_dimension)
            if self.intrinsic_dimension is None
            else float(self.intrinsic_dimension)
        )
        target, volume_diagnostics = estimate_uniform_volume_weights(
            local,
            intrinsic_dimension=dimension,
            k_neighbors=self.k_volume,
            winsor_quantile=self.winsor_quantile,
        )
        plan, transport_diagnostics = intrinsic_uniformization_plan(
            intrinsic,
            source,
            target,
            entropic_regularization=self.entropic_regularization,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_num_iter_max=self.sinkhorn_num_iter_max,
            sinkhorn_stop_threshold=self.sinkhorn_stop_threshold,
        )
        self.local_distances_ = local
        self.intrinsic_distances_ = intrinsic
        self.source_weights_ = source
        self.target_weights_ = target
        self.transport_plan_ = plan
        self.intrinsic_dimension_ = float(dimension)
        self.graph_diagnostics_ = graph_diagnostics
        self.volume_diagnostics_ = volume_diagnostics
        self.transport_diagnostics_ = transport_diagnostics
        self.fit_runtime_seconds_ = float(time.perf_counter() - started)
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before requesting anchor-indexed ranks.")

    def ranks_for_anchor(
        self,
        anchor_index: int,
        *,
        return_diagnostics: bool = False,
    ):
        self._check_fitted()
        scores, score_diagnostics = pooled_conditional_transport_wasserstein_scores(
            self.intrinsic_distances_,
            self.transport_plan_,
            self.source_weights_,
            anchor_index,
            anchor_pool_mass=self.anchor_pool_mass,
            minimum_anchor_rows=self.minimum_anchor_rows,
        )
        ranks = _weighted_cdf_levels(scores, self.source_weights_)
        diagnostics = {
            "method": "anchor-indexed ranks after one intrinsic uniformization",
            "anchor_index": int(anchor_index),
            "score": score_diagnostics,
            "fit_runtime_seconds": self.fit_runtime_seconds_,
            "anchor_runtime_seconds": score_diagnostics["runtime_seconds"],
            "intrinsic_dimension": self.intrinsic_dimension_,
            "graph": self.graph_diagnostics_,
            "volume": self.volume_diagnostics_,
            "transport": self.transport_diagnostics_,
            "raw_scores": scores,
            "ranks": ranks,
        }
        if return_diagnostics:
            return ranks, scores, diagnostics
        return ranks

    def ranks_for_anchors(
        self,
        anchor_indices: Sequence[int],
        *,
        return_diagnostics: bool = False,
    ):
        self._check_fitted()
        rank_output: Dict[int, np.ndarray] = {}
        score_output: Dict[int, np.ndarray] = {}
        diagnostic_output: Dict[int, Dict[str, Any]] = {}
        for anchor_index in anchor_indices:
            index = int(anchor_index)
            ranks, scores, diagnostics = self.ranks_for_anchor(
                index, return_diagnostics=True
            )
            rank_output[index] = ranks
            score_output[index] = scores
            diagnostic_output[index] = diagnostics
        if return_diagnostics:
            return rank_output, score_output, diagnostic_output
        return rank_output

    def diagnostics(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "method": "anchor-free intrinsic uniformization",
            "n_samples": int(len(self.local_distances_)),
            "intrinsic_dimension": self.intrinsic_dimension_,
            "graph": self.graph_diagnostics_,
            "volume": self.volume_diagnostics_,
            "transport": self.transport_diagnostics_,
            "anchor_pool_mass": self.anchor_pool_mass,
            "minimum_anchor_rows": self.minimum_anchor_rows,
            "fit_runtime_seconds": self.fit_runtime_seconds_,
            "local_distances": self.local_distances_,
            "intrinsic_distances": self.intrinsic_distances_,
            "source_weights": self.source_weights_,
            "estimated_volume_weights": self.target_weights_,
            "transport_plan": self.transport_plan_,
        }


class AnchoredIntrinsicUniformizer:
    """Distance-only estimator of anchored intrinsic quantile ranks."""

    def __init__(
        self,
        *,
        k_graph: int = 20,
        k_volume: int = 20,
        k_dimension: int = 15,
        intrinsic_dimension: Optional[float] = None,
        winsor_quantile: float = 0.01,
        ensure_connected: bool = True,
        maximum_graph_neighbors: Optional[int] = None,
        entropic_regularization: float = 0.0,
        sinkhorn_method: str = "sinkhorn_log",
        sinkhorn_num_iter_max: int = 20_000,
        sinkhorn_stop_threshold: float = 1e-9,
    ) -> None:
        self.k_graph = int(k_graph)
        self.k_volume = int(k_volume)
        self.k_dimension = int(k_dimension)
        self.intrinsic_dimension = intrinsic_dimension
        self.winsor_quantile = float(winsor_quantile)
        self.ensure_connected = bool(ensure_connected)
        self.maximum_graph_neighbors = maximum_graph_neighbors
        self.entropic_regularization = float(entropic_regularization)
        self.sinkhorn_method = str(sinkhorn_method)
        self.sinkhorn_num_iter_max = int(sinkhorn_num_iter_max)
        self.sinkhorn_stop_threshold = float(sinkhorn_stop_threshold)

    def fit_predict(
        self,
        local_distances: np.ndarray,
        *,
        intrinsic_distances: Optional[np.ndarray] = None,
        source_weights: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ):
        """Fit the estimator and return ``(center_index, ranks[, diagnostics])``."""
        started = time.perf_counter()
        local = _validate_distance_matrix(local_distances, "local_distances")
        n_samples = len(local)
        source = _validate_weights(source_weights, n_samples, "source_weights")

        if intrinsic_distances is None:
            intrinsic, graph_diagnostics = graph_geodesic_distances(
                local,
                k_neighbors=self.k_graph,
                ensure_connected=self.ensure_connected,
                maximum_neighbors=self.maximum_graph_neighbors,
            )
        else:
            intrinsic = _validate_distance_matrix(
                intrinsic_distances, "intrinsic_distances"
            )
            graph_diagnostics = {
                "source": "supplied intrinsic distance matrix",
                "requested_k_neighbors": None,
                "effective_k_neighbors": None,
            }

        dimension = (
            estimate_intrinsic_dimension(local, self.k_dimension)
            if self.intrinsic_dimension is None
            else float(self.intrinsic_dimension)
        )
        volume, volume_diagnostics = estimate_uniform_volume_weights(
            local,
            intrinsic_dimension=dimension,
            k_neighbors=self.k_volume,
            winsor_quantile=self.winsor_quantile,
        )
        center_index, medoid_objective = weighted_frechet_medoid(
            intrinsic, source
        )
        ranks, raw_scores, plan, adjusted_volume, transport_diagnostics = (
            anchored_transport_ranks(
                intrinsic,
                source,
                volume,
                center_index,
                entropic_regularization=self.entropic_regularization,
                sinkhorn_method=self.sinkhorn_method,
                sinkhorn_num_iter_max=self.sinkhorn_num_iter_max,
                sinkhorn_stop_threshold=self.sinkhorn_stop_threshold,
            )
        )

        diagnostics: Dict[str, Any] = {
            "method": "medoid-anchored intrinsic uniformization",
            "n_samples": int(n_samples),
            "intrinsic_dimension": float(dimension),
            "center_index": int(center_index),
            "graph": graph_diagnostics,
            "volume": volume_diagnostics,
            "transport": transport_diagnostics,
            "runtime_seconds": float(time.perf_counter() - started),
            "local_distances": local,
            "intrinsic_distances": intrinsic,
            "source_weights": source,
            "estimated_volume_weights": volume,
            "adjusted_volume_weights": adjusted_volume,
            "medoid_objective": medoid_objective,
            "transport_plan": plan,
            "raw_scores": raw_scores,
            "ranks": ranks,
        }
        if return_diagnostics:
            return center_index, ranks, diagnostics
        return center_index, ranks


__all__ = [
    "AnchorIndexedIntrinsicUniformizer",
    "AnchoredIntrinsicUniformizer",
    "anchored_transport_ranks",
    "conditional_transport_wasserstein_scores",
    "extend_anchored_ranks",
    "estimate_intrinsic_dimension",
    "estimate_uniform_volume_weights",
    "graph_geodesic_distances",
    "intrinsic_uniformization_plan",
    "pooled_conditional_transport_wasserstein_scores",
    "weighted_frechet_medoid",
]
