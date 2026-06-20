"""Train a real RBF SVM and plot its 2D boundary plus 3D decision surface."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from utils.data_generator import generate_ring_dataset
from utils.svm_utils import compute_decision_surface, grid_limits, make_decision_grid, train_svm


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    X, y = generate_ring_dataset(noise=0.08, random_seed=7)
    model = train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0)
    x_range, y_range = grid_limits(X)
    xx, yy, grid_points = make_decision_grid(x_range, y_range, resolution=160)
    Z = compute_decision_surface(model, grid_points, xx.shape)
    point_scores = model.decision_function(X)

    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    plot_2d(ax2d, X, y, model, xx, yy, Z)
    plot_3d(ax3d, X, y, model, xx, yy, Z, point_scores)

    fig.suptitle("Real RBF SVM: decision function visualization", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Note: z = f(x, y) visualizes the sklearn decision function. RBF does not map data directly to only 3D.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_dir / "rbf_decision_surface.png", dpi=160)
    plt.show()


def plot_2d(ax, X, y, model, xx, yy, Z) -> None:
    ax.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.25)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["gray", "gold", "gray"], linestyles=["--", "-", "--"])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="inner class", edgecolors="white")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", label="outer class", edgecolors="white")
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none", edgecolors="black", linewidths=1.4, label="support vectors")
    ax.set_title("2D boundary: f(x, y) = 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")


def plot_3d(ax, X, y, model, xx, yy, Z, point_scores) -> None:
    ax.plot_surface(xx, yy, Z, cmap="coolwarm", alpha=0.7, linewidth=0, antialiased=True)
    ax.contour(xx, yy, Z, levels=[0], colors="gold", linewidths=3, offset=0)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], point_scores[y == 0], c="tab:blue", edgecolors="white", s=35)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], point_scores[y == 1], c="tab:red", edgecolors="white", s=35)
    sv = model.support_vectors_
    sv_scores = model.decision_function(sv)
    ax.scatter(sv[:, 0], sv[:, 1], sv_scores, s=120, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.set_title("3D surface: z = f(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("decision score")


if __name__ == "__main__":
    main()
