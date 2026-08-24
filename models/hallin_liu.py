"""Known-geometry Hallin--Liu cap-grid benchmarks.

This module is an independent implementation of the two-step empirical
construction in Hallin and Liu (2024, arXiv:2410.15711).  It is deliberately
limited to geometries for which the analytic Riemannian distance and the
uniform cap contours are known: the unit two-sphere and the flat two-torus.

The conditional helper adds one clearly separated convention needed by the
AMQR out-of-sample benchmark.  The fitted Kantorovich plan is converted to a
source-observation score by taking the conditional expectation of the target
grid layer.  The original paper defines empirical regions by assigning grid
points back to observations; it does not propose this scalar OOS extension.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


Geometry = Literal["sphere", "flat_torus"]


def _normalise_weights(weights: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.shape != (int(n),):
        raise ValueError("weights must have one entry per source observation.")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("weights must be finite and strictly positive.")
    return values / np.sum(values)


def _validate_points(points: np.ndarray, geometry: Geometry) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    expected_dimension = 3 if geometry == "sphere" else 2
    if values.ndim != 2 or values.shape[1] != expected_dimension:
        raise ValueError(
            f"{geometry} points must have shape (n, {expected_dimension})."
        )
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("points must contain at least three finite observations.")
    if geometry == "sphere":
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("sphere points must be nonzero.")
        values = values / norms[:, None]
    else:
        values = np.mod(values, 2.0 * np.pi)
    return values


def uniform_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(int(n), 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def uniform_flat_torus(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * np.pi, size=(int(n), 2))


def sphere_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    inner = np.clip(np.asarray(left) @ np.asarray(right).T, -1.0, 1.0)
    return 0.5 * np.arccos(inner) ** 2


def signed_wrapped_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2.0 * np.pi) - np.pi


def flat_torus_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = signed_wrapped_difference(
        np.asarray(left)[:, None, 0], np.asarray(right)[None, :, 0]
    )
    second = signed_wrapped_difference(
        np.asarray(left)[:, None, 1], np.asarray(right)[None, :, 1]
    )
    return 0.5 * (first**2 + second**2)


def geometry_cost(
    left: np.ndarray, right: np.ndarray, geometry: Geometry
) -> np.ndarray:
    if geometry == "sphere":
        return sphere_cost(left, right)
    if geometry == "flat_torus":
        return flat_torus_cost(left, right)
    raise KeyError(geometry)


def factor_cap_grid(n: int) -> Tuple[int, int, int]:
    """Choose a balanced factorisation ``n=n_R*n_S+1``."""
    remainder = int(n) - 1
    divisors = [value for value in range(2, remainder + 1) if remainder % value == 0]
    if not divisors:
        raise ValueError(
            "The equal-weight cap construction requires composite n-1 "
            "(for example n=400)."
        )
    n_r = min(divisors, key=lambda value: abs(value - math.sqrt(remainder)))
    n_s = remainder // n_r
    if n_r > n_s:
        n_r, n_s = n_s, n_r
    return int(n_r), int(n_s), 1


def _tangent_basis(pole: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pole = np.asarray(pole, dtype=float)
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ pole)))]
    first = np.cross(pole, reference)
    first /= np.linalg.norm(first)
    second = np.cross(pole, first)
    second /= np.linalg.norm(second)
    return first, second


def structured_sphere_cap_grid(
    pole: np.ndarray, n_radial: int, n_contour: int, phase: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    pole = np.asarray(pole, dtype=float)
    pole = pole / np.linalg.norm(pole)
    first, second = _tangent_basis(pole)
    angle = float(phase) + 2.0 * np.pi * np.arange(int(n_contour)) / int(n_contour)
    tangent = (
        np.cos(angle)[:, None] * first[None, :]
        + np.sin(angle)[:, None] * second[None, :]
    )
    points: List[np.ndarray] = [pole]
    levels: List[float] = [0.0]
    for layer in range(1, int(n_radial) + 1):
        tau = layer / (int(n_radial) + 1.0)
        colatitude = math.acos(1.0 - 2.0 * tau)
        contour = math.cos(colatitude) * pole[None, :] + math.sin(colatitude) * tangent
        points.extend(contour)
        levels.extend([tau] * int(n_contour))
    return np.asarray(points), np.asarray(levels)


def _square_boundary_offsets(
    half_width: float, n_contour: int, phase: float
) -> np.ndarray:
    # Equally spaced perimeter locations, with a random phase to avoid giving
    # any one corner a deterministic advantage.
    position = (4.0 * np.arange(int(n_contour)) / int(n_contour) + phase) % 4.0
    output = np.empty((int(n_contour), 2), dtype=float)
    for index, value in enumerate(position):
        edge = int(math.floor(value))
        fraction = value - edge
        if edge == 0:
            output[index] = (-half_width + 2.0 * half_width * fraction, -half_width)
        elif edge == 1:
            output[index] = (half_width, -half_width + 2.0 * half_width * fraction)
        elif edge == 2:
            output[index] = (half_width - 2.0 * half_width * fraction, half_width)
        else:
            output[index] = (-half_width, half_width - 2.0 * half_width * fraction)
    return output


def structured_torus_cap_grid(
    pole: np.ndarray, n_radial: int, n_contour: int, phase: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Square probability contours on the flat product torus."""
    pole = np.mod(np.asarray(pole, dtype=float), 2.0 * np.pi)
    points: List[np.ndarray] = [pole]
    levels: List[float] = [0.0]
    for layer in range(1, int(n_radial) + 1):
        tau = layer / (int(n_radial) + 1.0)
        half_width = np.pi * math.sqrt(tau)
        offsets = _square_boundary_offsets(half_width, int(n_contour), float(phase))
        points.extend(np.mod(pole[None, :] + offsets, 2.0 * np.pi))
        levels.extend([tau] * int(n_contour))
    return np.asarray(points), np.asarray(levels)


def structured_cap_grid(
    pole: np.ndarray,
    n_radial: int,
    n_contour: int,
    geometry: Geometry,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if geometry == "sphere":
        return structured_sphere_cap_grid(pole, n_radial, n_contour, phase)
    if geometry == "flat_torus":
        return structured_torus_cap_grid(pole, n_radial, n_contour, phase)
    raise KeyError(geometry)


def _uniform_grid(
    n: int, geometry: Geometry, rng: np.random.Generator
) -> np.ndarray:
    if geometry == "sphere":
        return uniform_sphere(n, rng)
    if geometry == "flat_torus":
        return uniform_flat_torus(n, rng)
    raise KeyError(geometry)


def _assignment(cost: np.ndarray) -> Tuple[np.ndarray, float]:
    started = time.perf_counter()
    rows, columns = linear_sum_assignment(np.asarray(cost, dtype=float))
    assigned = np.empty(len(rows), dtype=int)
    assigned[rows] = columns
    return assigned, float(time.perf_counter() - started)


def unconditional_cap_ranks(
    points: np.ndarray,
    anchor_index: int,
    geometry: Geometry,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Two-step equal-weight empirical cap ranks on a known geometry."""
    values = _validate_points(points, geometry)
    n = len(values)
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < n:
        raise IndexError("anchor_index is outside the observed support.")
    n_radial, n_contour, n_zero = factor_cap_grid(n)

    preliminary = _uniform_grid(n, geometry, rng)
    first_cost = geometry_cost(values, preliminary, geometry)
    first_assignment, first_seconds = _assignment(first_cost)
    pole = preliminary[first_assignment[anchor_index]]

    phase = float(rng.uniform(0.0, 2.0 * np.pi if geometry == "sphere" else 4.0))
    regular, levels = structured_cap_grid(
        pole, n_radial, n_contour, geometry, phase
    )
    second_cost = geometry_cost(values, regular, geometry)
    second_assignment, second_seconds = _assignment(second_cost)
    ranks = levels[second_assignment]
    return ranks, {
        "implementation": "independent Hallin--Liu two-step equal-weight cap grid",
        "geometry": geometry,
        "n_radial": int(n_radial),
        "n_contour": int(n_contour),
        "n_zero": int(n_zero),
        "pole": pole,
        "first_assignment_seconds": first_seconds,
        "second_assignment_seconds": second_seconds,
        "runtime_seconds": first_seconds + second_seconds,
        "first_mean_cost": float(np.mean(first_cost[np.arange(n), first_assignment])),
        "second_mean_cost": float(np.mean(second_cost[np.arange(n), second_assignment])),
        "solver": "scipy.optimize.linear_sum_assignment",
        "solver_converged": True,
    }


def conditional_sphere_layer_scores(
    points: np.ndarray,
    source_weights: np.ndarray,
    anchor_index: int,
    rng: np.random.Generator,
    *,
    n_radial: int = 20,
    n_contour: int = 20,
    num_iter_max: int = 2_000_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Conditional two-step spherical cap fit plus an OOS-ready layer score.

    The first and second transports follow the conditional Kantorovich version
    of the known-geometry construction.  For source observation ``j``, the
    returned scalar is ``E[level(U) | Y_j]`` under the second fitted plan.  This
    last scalarisation is the benchmark's documented OOS extension.
    """
    started = time.perf_counter()
    values = _validate_points(points, "sphere")
    n = len(values)
    source = _normalise_weights(source_weights, n)
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < n:
        raise IndexError("anchor_index is outside the observed support.")
    if int(n_radial) < 2 or int(n_contour) < 4:
        raise ValueError("n_radial >= 2 and n_contour >= 4 are required.")

    grid_size = int(n_radial) * int(n_contour) + 1
    target = np.full(grid_size, 1.0 / grid_size)
    preliminary = uniform_sphere(grid_size, rng)
    first_cost = sphere_cost(values, preliminary)
    first_started = time.perf_counter()
    first_plan = ot.emd(source, target, first_cost, numItermax=int(num_iter_max))
    first_seconds = time.perf_counter() - first_started

    anchor_row = first_plan[anchor_index]
    maximum = float(np.max(anchor_row))
    candidates = np.flatnonzero(np.isclose(anchor_row, maximum, rtol=1e-10, atol=1e-15))
    if candidates.size > 1:
        anchor_cost = sphere_cost(values[[anchor_index]], preliminary[candidates])[0]
        pole_index = int(candidates[int(np.argmin(anchor_cost))])
    else:
        pole_index = int(candidates[0])
    pole = preliminary[pole_index]

    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    regular, levels = structured_sphere_cap_grid(
        pole, int(n_radial), int(n_contour), phase
    )
    second_cost = sphere_cost(values, regular)
    second_started = time.perf_counter()
    second_plan = ot.emd(source, target, second_cost, numItermax=int(num_iter_max))
    second_seconds = time.perf_counter() - second_started

    conditional = second_plan / source[:, None]
    raw_scores = conditional @ levels
    row_error = float(np.max(np.abs(second_plan.sum(axis=1) - source)))
    column_error = float(np.max(np.abs(second_plan.sum(axis=0) - target)))
    return np.clip(raw_scores, 0.0, 1.0), {
        "implementation": "conditional Hallin--Liu grid with expected-layer OOS extension",
        "geometry": "sphere",
        "n_radial": int(n_radial),
        "n_contour": int(n_contour),
        "grid_size": int(grid_size),
        "pole": pole,
        "pole_grid_index": int(pole_index),
        "first_transport_cost": float(np.sum(first_plan * first_cost)),
        "second_transport_cost": float(np.sum(second_plan * second_cost)),
        "first_row_marginal_max_error": float(
            np.max(np.abs(first_plan.sum(axis=1) - source))
        ),
        "first_column_marginal_max_error": float(
            np.max(np.abs(first_plan.sum(axis=0) - target))
        ),
        "row_marginal_max_error": row_error,
        "column_marginal_max_error": column_error,
        "first_transport_seconds": float(first_seconds),
        "second_transport_seconds": float(second_seconds),
        "runtime_seconds": float(time.perf_counter() - started),
        "solver": "POT ot.emd",
        "solver_converged": bool(max(row_error, column_error) <= 1e-8),
        "score_definition": "source-conditional expectation of target cap layer",
    }


__all__ = [
    "conditional_sphere_layer_scores",
    "factor_cap_grid",
    "flat_torus_cost",
    "signed_wrapped_difference",
    "sphere_cost",
    "structured_sphere_cap_grid",
    "structured_torus_cap_grid",
    "unconditional_cap_ranks",
    "uniform_flat_torus",
    "uniform_sphere",
]
