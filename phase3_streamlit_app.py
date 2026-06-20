"""Interactive Streamlit and Plotly SVM kernel demo."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data_generator import generate_ring_dataset
from utils.svm_utils import compute_decision_surface, grid_limits, make_decision_grid, train_svm


st.set_page_config(page_title="Interactive SVM Kernel Trick 3D Demo", layout="wide")

DYNAMIC_GAMMA_STEPS = 15
HEADER_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "svm-tech-header.png"

TEXT = {
    "en": {
        "language_toggle": "中文",
        "title": "Interactive SVM Kernel Trick 3D Demo",
        "kernel": "kernel",
        "gamma": "gamma",
        "degree": "degree",
        "noise": "noise",
        "number_of_points": "number_of_points",
        "random_seed": "random_seed",
        "grid_resolution": "grid_resolution",
        "concept": "Concept",
        "concept_body": (
            "2D circular data cannot be separated by a straight line. Kernel SVMs learn nonlinear "
            "decision boundaries; the RBF kernel does this through similarity to support vectors."
        ),
        "support_vectors": "Support vectors",
        "training_accuracy": "Training accuracy",
        "kernel_metric": "Kernel",
        "boundary_2d": "2D Decision Boundary",
        "surface_3d": "3D Decision Function Surface",
        "dynamic_effect": "Dynamic Model Effect",
        "teaching_notes": "Teaching Notes",
        "caption": (
            "Important note: z = x^2 + y^2 is a classroom feature-map visualization. "
            "The RBF decision surface here is z = f(x, y), not the full RBF feature space."
        ),
        "decision_function": "decision function",
        "zero_plane": "z = 0",
        "inner_class": "inner class",
        "outer_class": "outer class",
        "play": "Play",
        "z_axis": "f(x, y)",
        "small_gamma": "Gamma is small: the boundary is smoother and each point has wider influence.",
        "large_gamma": "Gamma is large: the boundary becomes very flexible and may overfit.",
        "small_c": "C is small: the model allows more mistakes to keep a wider margin.",
        "large_c": "C is large: the model tries harder to classify training data correctly.",
        "default_note": "Try changing C and gamma to see the margin and boundary complexity move.",
    },
    "zh": {
        "language_toggle": "中文",
        "title": "互動式 SVM 核技巧 3D 示範",
        "kernel": "核函數",
        "gamma": "gamma",
        "degree": "多項式次方",
        "noise": "雜訊",
        "number_of_points": "資料點數量",
        "random_seed": "隨機種子",
        "grid_resolution": "網格解析度",
        "concept": "概念",
        "concept_body": "2D 圓形資料無法用一條直線分開。Kernel SVM 會學到非線性決策邊界；RBF 核透過資料點之間的相似度完成這件事。",
        "support_vectors": "支持向量",
        "training_accuracy": "訓練準確率",
        "kernel_metric": "核函數",
        "boundary_2d": "2D 決策邊界",
        "surface_3d": "3D 決策函數曲面",
        "dynamic_effect": "動態模型效果",
        "teaching_notes": "教學提示",
        "caption": "重要說明：z = x^2 + y^2 是課堂上的特徵映射視覺化。這裡的 RBF 決策曲面是 z = f(x, y)，不是完整的 RBF 特徵空間。",
        "decision_function": "決策函數",
        "zero_plane": "z = 0",
        "inner_class": "內圈類別",
        "outer_class": "外圈類別",
        "play": "播放",
        "z_axis": "f(x, y)",
        "small_gamma": "Gamma 較小：邊界較平滑，每個點的影響範圍較大。",
        "large_gamma": "Gamma 較大：邊界更有彈性，但可能過度擬合。",
        "small_c": "C 較小：模型允許較多錯誤，以保留較寬的 margin。",
        "large_c": "C 較大：模型會更努力把訓練資料分類正確。",
        "default_note": "試著調整 C 和 gamma，觀察 margin 與決策邊界複雜度如何改變。",
    },
}


@st.cache_data
def cached_dataset(n_points: int, noise: float, random_seed: int):
    n_inner = max(10, int(n_points * 0.44))
    n_outer = max(10, n_points - n_inner)
    return generate_ring_dataset(n_inner=n_inner, n_outer=n_outer, noise=noise, random_seed=random_seed)


@st.cache_data
def cached_surface(X, y, kernel: str, C: float, gamma: float, degree: int, resolution: int):
    model = train_svm(np.asarray(X), np.asarray(y), kernel=kernel, C=C, gamma=gamma, degree=degree)
    x_range, y_range = grid_limits(np.asarray(X))
    xx, yy, grid_points = make_decision_grid(x_range, y_range, resolution=resolution)
    Z = compute_decision_surface(model, grid_points, xx.shape)
    point_scores = model.decision_function(np.asarray(X))
    return model, xx, yy, Z, point_scores


@st.cache_data
def cached_dynamic_surfaces(X, y, kernel: str, C: float, gamma: float, degree: int, resolution: int):
    x_range, y_range = grid_limits(np.asarray(X))
    xx, yy, grid_points = make_decision_grid(x_range, y_range, resolution=resolution)
    frames = []
    for step in np.linspace(0.2, 1.0, DYNAMIC_GAMMA_STEPS):
        frame_C = max(0.1, C * step)
        frame_gamma = gamma
        if kernel in {"rbf", "poly", "sigmoid"}:
            frame_gamma = max(0.01, gamma * step)
        model = train_svm(np.asarray(X), np.asarray(y), kernel=kernel, C=frame_C, gamma=frame_gamma, degree=degree)
        Z = compute_decision_surface(model, grid_points, xx.shape)
        point_scores = model.decision_function(np.asarray(X))
        sv = model.support_vectors_
        frames.append(
            {
                "label": f"C={frame_C:.2f}, gamma={frame_gamma:.2f}",
                "Z": Z,
                "point_scores": point_scores,
                "support_vectors": sv,
                "support_scores": model.decision_function(sv),
            }
        )
    return xx, yy, frames


def main() -> None:
    with st.sidebar:
        use_chinese = st.toggle("English / 中文", value=False)
        text = TEXT["zh" if use_chinese else "en"]
        kernel = st.selectbox(text["kernel"], ["linear", "poly", "rbf", "sigmoid"], index=2)
        C = st.slider("C", 0.1, 100.0, 10.0, 0.1)
        gamma = 1.0
        if kernel in {"rbf", "poly", "sigmoid"}:
            gamma = st.slider(text["gamma"], 0.01, 10.0, 1.0, 0.01)
        degree = 3
        if kernel == "poly":
            degree = st.slider(text["degree"], 2, 6, 3, 1)
        noise = st.slider(text["noise"], 0.0, 0.5, 0.08, 0.01)
        n_points = st.slider(text["number_of_points"], 40, 300, 120, 10)
        random_seed = st.number_input(text["random_seed"], value=7, step=1)
        resolution = st.slider(text["grid_resolution"], 50, 150, 80, 10)

    render_header_image()
    st.title(text["title"])

    X, y = cached_dataset(n_points, noise, int(random_seed))
    model, xx, yy, Z, point_scores = cached_surface(X, y, kernel, C, gamma, degree, resolution)
    accuracy = float(model.score(X, y))

    st.subheader(text["concept"])
    st.write(text["concept_body"])

    metrics = st.columns(4)
    metrics[0].metric(text["support_vectors"], len(model.support_))
    metrics[1].metric(text["training_accuracy"], f"{accuracy:.3f}")
    metrics[2].metric(text["kernel_metric"], kernel)
    metrics[3].metric("C", f"{C:.1f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(text["boundary_2d"])
        st.plotly_chart(make_2d_figure(X, y, model, xx, yy, Z, text), use_container_width=True)
    with col_right:
        st.subheader(text["surface_3d"])
        st.plotly_chart(make_3d_figure(X, y, model, xx, yy, Z, point_scores, text), use_container_width=True)

    st.subheader(text["dynamic_effect"])
    dynamic_resolution = min(resolution, 90)
    dyn_xx, dyn_yy, dynamic_frames = cached_dynamic_surfaces(X, y, kernel, C, gamma, degree, dynamic_resolution)
    st.plotly_chart(make_dynamic_3d_figure(X, y, dyn_xx, dyn_yy, dynamic_frames, text), use_container_width=True)

    st.subheader(text["teaching_notes"])
    for note in teaching_notes(C, gamma, kernel, text):
        st.info(note)
    st.caption(text["caption"])


def render_header_image() -> None:
    image_data = base64.b64encode(HEADER_IMAGE_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <img
            src="data:image/png;base64,{image_data}"
            style="width: 100%; height: 300px; object-fit: cover; border-radius: 8px; display: block;"
            alt="SVM technology header"
        />
        """,
        unsafe_allow_html=True,
    )


def make_2d_figure(X, y, model, xx, yy, Z, text):
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale="RdBu", opacity=0.35, showscale=False, contours_coloring="heatmap"))
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=0, end=0, size=1), line=dict(color="gold", width=4), showscale=False, name="f = 0"))
    for level, name in [(-1, "f = -1"), (1, "f = +1")]:
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=level, end=level, size=1), line=dict(color="gray", width=2, dash="dash"), showscale=False, name=name))
    add_points_2d(fig, X, y, text)
    sv = model.support_vectors_
    fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=2)), name=text["support_vectors"]))
    fig.update_layout(height=560, xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x", legend=dict(orientation="h"))
    return fig


def make_3d_figure(X, y, model, xx, yy, Z, point_scores, text):
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xx, y=yy, z=Z, colorscale="RdBu", opacity=0.82, showscale=False, name=text["decision_function"]))
    fig.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(Z), opacity=0.18, colorscale=[[0, "gold"], [1, "gold"]], showscale=False, name=text["zero_plane"]))
    add_points_3d(fig, X, y, point_scores, text)
    sv = model.support_vectors_
    sv_scores = model.decision_function(sv)
    fig.add_trace(go.Scatter3d(x=sv[:, 0], y=sv[:, 1], z=sv_scores, mode="markers", marker=dict(size=7, color="white", line=dict(color="black", width=4)), name=text["support_vectors"]))
    fig.update_layout(height=560, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=text["z_axis"]), legend=dict(orientation="h"))
    return fig


def make_dynamic_3d_figure(X, y, xx, yy, dynamic_frames, text):
    first = dynamic_frames[0]
    fig = go.Figure(
        data=dynamic_frame_traces(X, y, xx, yy, first, text),
        frames=[
            go.Frame(data=dynamic_frame_traces(X, y, xx, yy, frame, text), name=str(index))
            for index, frame in enumerate(dynamic_frames)
        ],
    )
    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame["label"],
            "method": "animate",
        }
        for index, frame in enumerate(dynamic_frames)
    ]
    fig.update_layout(
        height=560,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=text["z_axis"]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": text["play"],
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 650, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 250},
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[{"active": 0, "steps": slider_steps, "x": 0.1, "len": 0.86}],
        legend=dict(orientation="h"),
    )
    return fig


def dynamic_frame_traces(X, y, xx, yy, frame, text):
    sv = frame["support_vectors"]
    return [
        go.Surface(x=xx, y=yy, z=frame["Z"], colorscale="RdBu", opacity=0.82, showscale=False, name=text["decision_function"]),
        go.Surface(x=xx, y=yy, z=np.zeros_like(frame["Z"]), opacity=0.18, colorscale=[[0, "gold"], [1, "gold"]], showscale=False, name=text["zero_plane"]),
        go.Scatter3d(x=X[y == 0, 0], y=X[y == 0, 1], z=frame["point_scores"][y == 0], mode="markers", marker=dict(size=4, color="royalblue"), name=text["inner_class"]),
        go.Scatter3d(x=X[y == 1, 0], y=X[y == 1, 1], z=frame["point_scores"][y == 1], mode="markers", marker=dict(size=4, color="crimson"), name=text["outer_class"]),
        go.Scatter3d(x=sv[:, 0], y=sv[:, 1], z=frame["support_scores"], mode="markers", marker=dict(size=7, color="white", line=dict(color="black", width=4)), name=text["support_vectors"]),
    ]


def add_points_2d(fig, X, y, text) -> None:
    fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode="markers", marker=dict(size=8, color="royalblue", line=dict(color="white", width=1)), name=text["inner_class"]))
    fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode="markers", marker=dict(size=8, color="crimson", line=dict(color="white", width=1)), name=text["outer_class"]))


def add_points_3d(fig, X, y, scores, text) -> None:
    fig.add_trace(go.Scatter3d(x=X[y == 0, 0], y=X[y == 0, 1], z=scores[y == 0], mode="markers", marker=dict(size=4, color="royalblue"), name=text["inner_class"]))
    fig.add_trace(go.Scatter3d(x=X[y == 1, 0], y=X[y == 1, 1], z=scores[y == 1], mode="markers", marker=dict(size=4, color="crimson"), name=text["outer_class"]))


def teaching_notes(C: float, gamma: float, kernel: str, text) -> list[str]:
    notes = []
    if kernel in {"rbf", "poly", "sigmoid"}:
        if gamma < 0.2:
            notes.append(text["small_gamma"])
        if gamma > 3:
            notes.append(text["large_gamma"])
    if C < 1:
        notes.append(text["small_c"])
    if C > 20:
        notes.append(text["large_c"])
    if not notes:
        notes.append(text["default_note"])
    return notes


if __name__ == "__main__":
    main()
