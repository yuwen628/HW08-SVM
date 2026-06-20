"""Dataset helpers for circular SVM examples."""

from __future__ import annotations

import numpy as np


def generate_ring_dataset(
    n_inner: int = 35,
    n_outer: int = 45,
    inner_radius_range: tuple[float, float] = (0.0, 1.0),
    outer_radius_range: tuple[float, float] = (1.6, 2.5),
    noise: float = 0.08,
    random_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two classes: inner blue points and an outer red ring."""
    rng = np.random.default_rng(random_seed)

    inner_angles = rng.uniform(0, 2 * np.pi, n_inner)
    inner_radii = np.sqrt(rng.uniform(inner_radius_range[0] ** 2, inner_radius_range[1] ** 2, n_inner))
    outer_angles = rng.uniform(0, 2 * np.pi, n_outer)
    outer_radii = rng.uniform(outer_radius_range[0], outer_radius_range[1], n_outer)

    inner = np.column_stack((inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)))
    outer = np.column_stack((outer_radii * np.cos(outer_angles), outer_radii * np.sin(outer_angles)))

    if noise > 0:
        inner += rng.normal(0, noise, inner.shape)
        outer += rng.normal(0, noise, outer.shape)

    X = np.vstack((inner, outer))
    y = np.concatenate((np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)))
    return X, y

