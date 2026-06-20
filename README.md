# SVM Kernel Trick 3D Interactive Demo V2

## Project Overview

This project is a classroom-oriented demonstration of the Support Vector Machine
kernel trick. It contains:

- A Manim concept animation for lifting circular 2D data into 3D.
- A real sklearn RBF SVM decision function visualization.
- An interactive Streamlit and Plotly app for exploring kernel parameters.
- A dynamic model effect that animates the decision surface from a simpler model
  toward the selected sidebar parameters.

## Educational Story

The dataset has blue points near the origin and red points in an outer ring. In
the original 2D plane, a straight line cannot separate the two classes. A feature
mapping can make the same data linearly separable in a higher-dimensional space.

## Phase 1: Manim Kernel Trick Animation

File: `phase1_manim_kernel_trick.py`

The animation uses the teaching map:

```text
phi(x, y) = (x, y, x^2 + y^2)
```

Inner blue points stay low, outer red points lift higher, and a horizontal plane
separates them in 3D. Projecting the plane back to 2D gives a circular decision
boundary.

## Phase 2: Real RBF SVM Decision Surface

File: `phase2_rbf_decision_surface.py`

This script trains an actual sklearn `SVC(kernel="rbf")`, draws the 2D decision
boundary `f(x, y) = 0`, margin contours `f(x, y) = -1` and `f(x, y) = +1`, and a
3D surface `z = f(x, y)`.

Support vectors are highlighted in both plots.

## Phase 3: Interactive Streamlit Demo

File: `phase3_streamlit_app.py`

The app lets students adjust:

- kernel: `linear`, `poly`, `rbf`, `sigmoid`
- `C`
- `gamma`
- polynomial `degree`
- noise
- number of points
- random seed
- grid resolution

The 2D boundary, 3D decision function surface, support vector count, and teaching
notes update interactively.

V2 adds a **Dynamic Model Effect** section. It creates several intermediate SVM
models by gradually increasing `C` and, for nonlinear kernels, `gamma`, then uses
Plotly animation controls to show how the 3D decision function surface changes.

## Installation

```bash
pip install -r requirements.txt
```

If Manim fails to install on Windows with a `moderngl` or `glcontext` compiler
error, use Python 3.11 or 3.12 for the Manim environment, or install Microsoft
C++ Build Tools and rerun the command. The sklearn, matplotlib, Streamlit, and
Plotly parts can run without Manim.

## Run Commands

```bash
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
python phase2_rbf_decision_surface.py
streamlit run phase3_streamlit_app.py
```

Run the commands from the `WH8-SVM-v2` directory.

## Important Mathematical Note

The mapping `z = x^2 + y^2` is used as a visual and educational feature mapping
to explain why nonlinear data can become linearly separable in a
higher-dimensional feature space.

A real RBF kernel does not explicitly map data to only 3D. It corresponds to a
high-dimensional or infinite-dimensional feature space. Therefore, the RBF
decision surface shown in Phase 2 and Phase 3 visualizes the decision function
`f(x, y)`, not the full feature space itself.

## Teaching Suggestions

- Start with Phase 1 to build geometric intuition.
- Use Phase 2 to show that sklearn is training a real RBF SVM.
- Use Phase 3 live in class and ask students to predict what changes when `C`
  or `gamma` moves.
- Compare small gamma with large gamma to discuss smoothness and overfitting.
- Compare small C with large C to discuss soft margins.
