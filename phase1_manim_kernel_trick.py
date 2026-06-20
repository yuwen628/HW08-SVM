"""Manim animation for the SVM kernel trick intuition.

This scene uses the teaching map phi(x, y) = (x, y, x^2 + y^2). It is not a
claim that the RBF kernel maps data into only three dimensions.
"""

from __future__ import annotations

import numpy as np
from manim import *


def generate_points(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inner_angles = rng.uniform(0, TAU, 35)
    inner_radii = np.sqrt(rng.uniform(0.0, 1.0, 35))
    outer_angles = rng.uniform(0, TAU, 45)
    outer_radii = rng.uniform(1.6, 2.5, 45)
    inner = np.column_stack((inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)))
    outer = np.column_stack((outer_radii * np.cos(outer_angles), outer_radii * np.sin(outer_angles)))
    X = np.vstack((inner, outer))
    y = np.concatenate((np.zeros(len(inner), dtype=int), np.ones(len(outer), dtype=int)))
    return X, y


def lifted_z(X: np.ndarray) -> np.ndarray:
    return np.sum(X**2, axis=1)


class SVMKernelTrick3D(ThreeDScene):
    def construct(self) -> None:
        self.camera.background_color = "#111111"
        X, y = generate_points()
        c = 1.35

        title = Text("SVM Kernel Trick: From 2D to 3D", font_size=36)
        subtitle = Text("Nonlinear in 2D, linear in feature space.", font_size=22)
        subtitle.next_to(title, DOWN, buff=0.25)
        opening = VGroup(title, subtitle).to_edge(UP)
        self.play(Write(opening))
        self.wait(1)
        self.play(FadeOut(opening))

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 7, 1],
            x_length=6.5,
            y_length=6.5,
            z_length=4.0,
        )
        labels = axes.get_axis_labels(Tex("$x$"), Tex("$y$"), Tex("$z$"))
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

        points_2d = self.make_points(axes, X, y, np.zeros(len(X)))
        self.play(Create(axes), Write(labels), FadeIn(points_2d), run_time=1.5)
        no_line = Text("No straight line can separate them in 2D.", font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(no_line)
        self.play(Write(no_line))
        self.wait(1)
        self.play(FadeOut(no_line))
        self.remove_fixed_in_frame_mobjects(no_line)

        formula = MathTex(r"\phi(x,y)=(x,y,x^2+y^2)", font_size=34).to_corner(UL)
        lift_note = Text("Lift each point by its distance from the origin.", font_size=22).next_to(formula, DOWN)
        self.add_fixed_in_frame_mobjects(formula, lift_note)
        self.play(Write(formula), FadeIn(lift_note))

        points_3d = self.make_points(axes, X, y, lifted_z(X))
        self.play(Transform(points_2d, points_3d), run_time=2.6)
        self.wait(0.5)

        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2.6, 2.6],
            v_range=[-2.6, 2.6],
            resolution=(36, 36),
        )
        paraboloid.set_style(fill_color=BLUE_E, fill_opacity=0.22, stroke_color=BLUE_D, stroke_width=0.25)
        self.play(Create(paraboloid), run_time=1.6)

        plane = Surface(
            lambda u, v: axes.c2p(u, v, c),
            u_range=[-2.8, 2.8],
            v_range=[-2.8, 2.8],
            resolution=(2, 2),
        )
        plane.set_style(fill_color=YELLOW, fill_opacity=0.35, stroke_color=YELLOW, stroke_width=0.6)
        plane_label = Text("Hyperplane in feature space", font_size=22).to_corner(UR)
        self.add_fixed_in_frame_mobjects(plane_label)
        self.play(Create(plane), FadeIn(plane_label), run_time=1.2)

        projection = MathTex(r"z=c,\ z=x^2+y^2\ \Rightarrow\ x^2+y^2=c", font_size=30).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(projection)
        circle = ParametricFunction(
            lambda t: axes.c2p(np.sqrt(c) * np.cos(t), np.sqrt(c) * np.sin(t), 0),
            t_range=[0, TAU],
            color=YELLOW,
            stroke_width=6,
        )
        self.play(Write(projection), Create(circle), run_time=1.5)

        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        self.remove_fixed_in_frame_mobjects(formula, lift_note, plane_label, projection)
        self.play(
            FadeOut(VGroup(axes, labels, points_2d, paraboloid, plane, circle)),
            FadeOut(formula),
            FadeOut(lift_note),
            FadeOut(plane_label),
            FadeOut(projection),
        )

        summary = VGroup(
            Text("In 3D: linear hyperplane", font_size=30),
            Text("In 2D: nonlinear decision boundary", font_size=30),
            Text("This is the intuition behind the kernel trick.", font_size=28),
        ).arrange(DOWN, buff=0.35)
        self.play(FadeIn(summary, shift=UP))
        self.wait(2)

    def make_points(self, axes: ThreeDAxes, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> VGroup:
        points = VGroup()
        for (x_val, y_val), label, z_val in zip(X, y, z):
            color = BLUE if label == 0 else RED
            points.add(Dot3D(axes.c2p(x_val, y_val, z_val), radius=0.055, color=color))
        return points

