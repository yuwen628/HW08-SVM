"""SVM helpers shared by static and interactive visualizations."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: float | str = 1.0,
    degree: int = 3,
) -> SVC:
    """Train an sklearn SVC using the requested kernel parameters."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model.fit(X, y)
    return model


def make_decision_grid(
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D grid and flattened coordinate array."""
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, grid_points


def compute_decision_surface(model: SVC, grid_points: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Evaluate model decision scores and reshape them to the grid."""
    scores = model.decision_function(grid_points)
    return scores.reshape(grid_shape)


def grid_limits(X: np.ndarray, padding: float = 0.7) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded plotting limits for a dataset."""
    x_min, y_min = X.min(axis=0) - padding
    x_max, y_max = X.max(axis=0) + padding
    return (float(x_min), float(x_max)), (float(y_min), float(y_max))

