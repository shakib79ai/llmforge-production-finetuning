#!/usr/bin/env python
"""Render the llmforge pipeline as an animated GIF (matplotlib, no ffmpeg required)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

INK = "#0e1a1f"
PANEL = "#142329"
BORDER = "#2a3d42"
TEXT = "#e7efee"
TEXT_DIM = "#9db0b4"
TEXT_FAINT = "#647579"
ACCENT = "#e2963c"
OK = "#4fa98a"
IDLE = "#64777c"
STATUS_COLOR = {"ok": OK, "partial": ACCENT, "idle": IDLE}

STAGES = [
    dict(name="DATA PREP", sub="verified", x=70, y=115, status="ok"),
    dict(name="SFT TRAIN", sub="needs GPU", x=242, y=72, status="idle"),
    dict(name="EVALUATION", sub="metrics tested", x=414, y=158, status="partial"),
    dict(name="TRACKING", sub="not run", x=586, y=72, status="idle"),
    dict(name="MERGE / QUANTIZE", sub="not run", x=758, y=158, status="idle"),
    dict(name="DEPLOY", sub="verified", x=930, y=115, status="ok"),
]

BEZIER_A = ((70, 115), (220, 40), (340, 190), (500, 115))
BEZIER_B = ((500, 115), (660, 40), (780, 40), (930, 115))


def cubic_bezier(t: float, p0, p1, p2, p3) -> tuple[float, float]:
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def path_point(t: float) -> tuple[float, float]:
    if t < 0.5:
        return cubic_bezier(t * 2, *BEZIER_A)
    return cubic_bezier((t - 0.5) * 2, *BEZIER_B)


PATH_SAMPLES = np.array([path_point(t) for t in np.linspace(0, 1, 400)])
NODE_T = [(s["x"] - 70) / (930 - 70) for s in STAGES]
N_FRAMES = 240


def build_figure():
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1000)
    ax.set_ylim(250, 0)
    ax.axis("off")

    ax.plot(PATH_SAMPLES[:, 0], PATH_SAMPLES[:, 1], color=BORDER, linewidth=2, zorder=1)
    ax.text(500, 16, "LLMFORGE — PIPELINE FLOW", ha="center", fontsize=10,
             family="monospace", color=TEXT_DIM, weight="bold")

    node_glows, node_rings = [], []
    for s in STAGES:
        color = STATUS_COLOR[s["status"]]
        glow = plt.Circle((s["x"], s["y"]), 15, color=color, alpha=0.0, zorder=2)
        ring = plt.Circle((s["x"], s["y"]), 15, facecolor=PANEL, edgecolor=color, linewidth=2.2, zorder=3)
        ax.add_patch(glow)
        ax.add_patch(ring)
        node_glows.append(glow)
        node_rings.append(ring)

        above = s["y"] < 100
        label_y = s["y"] - 32 if above else s["y"] + 32
        sub_y = s["y"] - 46 if above else s["y"] + 46
        ax.text(s["x"], label_y, s["name"], ha="center", va="center",
                 fontsize=8.5, family="monospace", color=TEXT, zorder=4)
        ax.text(s["x"], sub_y, s["sub"], ha="center", va="center",
                 fontsize=7, family="monospace", color=TEXT_FAINT, zorder=4)

    legend_items = [("verified this session", OK), ("partially exercised", ACCENT), ("not run", IDLE)]
    item_widths = [18 + len(label) * 6.4 + 34 for label, _ in legend_items]
    lx = 500 - sum(item_widths) / 2
    for (label, color), width in zip(legend_items, item_widths):
        ax.scatter([lx], [236], s=40, color=color, zorder=4)
        ax.text(lx + 14, 236, label, ha="left", va="center", fontsize=7.5,
                 family="monospace", color=TEXT_FAINT, zorder=4)
        lx += width

    pulse_halo, = ax.plot([], [], "o", color=ACCENT, markersize=16, alpha=0.3, zorder=4)
    pulse_core, = ax.plot([], [], "o", color=ACCENT, markersize=6, zorder=5)
    trail_line, = ax.plot([], [], color=ACCENT, linewidth=2, alpha=0.8, zorder=4)

    return fig, ax, node_glows, pulse_halo, pulse_core, trail_line


def make_animation():
    fig, ax, node_glows, pulse_halo, pulse_core, trail_line = build_figure()

    def update(frame):
        t = (frame / N_FRAMES) % 1.0
        x, y = path_point(t)
        pulse_core.set_data([x], [y])
        pulse_halo.set_data([x], [y])

        trail_t = np.linspace(max(0.0, t - 0.1), t, 30)
        trail_xy = np.array([path_point(tt) for tt in trail_t])
        trail_line.set_data(trail_xy[:, 0], trail_xy[:, 1])

        for i, s in enumerate(STAGES):
            d = abs(t - NODE_T[i])
            d = min(d, 1 - d)
            amp = 1.0 if s["status"] != "idle" else 0.4
            alpha = amp * np.exp(-(d**2) / (2 * 0.035**2))
            node_glows[i].set_alpha(min(alpha, 0.9))
            node_glows[i].set_radius(15 + 10 * alpha)

        return [pulse_core, pulse_halo, trail_line, *node_glows]

    return fig, FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / 30, blit=False)


def main() -> None:
    fig, anim = make_animation()
    output = Path(__file__).parent / "workflow_visualization.gif"
    anim.save(output, writer=PillowWriter(fps=30))
    print(f"Saved animation to {output}")


if __name__ == "__main__":
    main()
