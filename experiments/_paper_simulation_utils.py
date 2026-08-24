"""Shared deterministic geometry helpers for the paper experiments.

This module contains only simulation-oracle geometry and deterministic tuning
rules.  The AMQR estimators never receive these parameterizations: the paper
entry-point scripts pass them only to data generation, oracle evaluation, and
plotting.  Keeping the helpers here prevents current experiments from importing
retired Legacy/hard-anchor experiment drivers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sphere_geodesic(
    left: np.ndarray, right: np.ndarray | None = None
) -> np.ndarray:
    """Great-circle distance on the unit sphere (oracle use only)."""
    left = np.asarray(left, dtype=float)
    right = left if right is None else np.asarray(right, dtype=float)
    return np.arccos(np.clip(left @ right.T, -1.0, 1.0))


def wrapped_angle_distance(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Pairwise shortest angular distance on the circle."""
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    difference = np.abs(left[:, None] - right[None, :])
    return np.minimum(difference, 2.0 * np.pi - difference)


def _wavy_curve_table(
    n_grid: int = 40_001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    parameter = np.linspace(0.0, 2.0 * np.pi, int(n_grid))
    radius = 1.0 + 0.18 * np.cos(3.0 * parameter)
    points = np.column_stack(
        [
            radius * np.cos(parameter),
            radius * np.sin(parameter),
            0.28 * np.sin(2.0 * parameter),
        ]
    )
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return parameter, points, cumulative_length, float(cumulative_length[-1])


CURVE_PARAMETER, CURVE_POINTS, CURVE_ARC, CURVE_LENGTH = _wavy_curve_table()


def embed_wavy_curve(arc_angle: np.ndarray) -> np.ndarray:
    """Embed intrinsic arc-angle coordinates as a closed wavy curve in R3."""
    target_length = (
        np.mod(np.asarray(arc_angle, dtype=float), 2.0 * np.pi)
        / (2.0 * np.pi)
        * CURVE_LENGTH
    )
    parameter = np.interp(target_length, CURVE_ARC, CURVE_PARAMETER)
    radius = 1.0 + 0.18 * np.cos(3.0 * parameter)
    return np.column_stack(
        [
            radius * np.cos(parameter),
            radius * np.sin(parameter),
            0.28 * np.sin(2.0 * parameter),
        ]
    )


def torus_geodesic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Product geodesic on the flat two-torus."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    first = wrapped_angle_distance(left[:, 0], right[:, 0])
    second = wrapped_angle_distance(left[:, 1], right[:, 1])
    return np.sqrt(first ** 2 + second ** 2)


def torus_display(angle: np.ndarray) -> np.ndarray:
    """Three-dimensional display embedding of flat-torus coordinates."""
    angle = np.asarray(angle, dtype=float)
    major_radius = 1.65
    minor_radius = 0.62
    first, second = angle[:, 0], angle[:, 1]
    return np.column_stack(
        [
            (major_radius + minor_radius * np.cos(second)) * np.cos(first),
            (major_radius + minor_radius * np.cos(second)) * np.sin(first),
            minor_radius * np.sin(second),
        ]
    )


def adaptive_neighbors(
    n_samples: int, true_dimension: int
) -> Tuple[int, int, int]:
    """Deterministic neighbourhood rules reported in Section 4.1."""
    log_n = np.log(max(int(n_samples), 3))
    dimension_neighbors = int(np.clip(round(2.2 * log_n), 8, 30))
    if int(true_dimension) == 1:
        graph_neighbors = int(np.clip(round(1.45 * log_n), 6, 12))
    else:
        graph_neighbors = int(np.clip(round(2.15 * log_n), 9, 20))
    volume_neighbors = int(np.clip(round(np.sqrt(n_samples)), 10, 40))
    return dimension_neighbors, graph_neighbors, volume_neighbors
