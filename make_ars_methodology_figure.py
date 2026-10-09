#!/usr/bin/env python3
"""Generate a text-only, full-width ARS / MPPI / MJX methodology figure.

Usage (only matplotlib is required):
    python make_ars_methodology_figure.py
    python make_ars_methodology_figure.py --overwrite

Outputs: methodology_figures/ars_mppi_mjx_methodology_wide.{pdf,svg,png}
PDF and SVG are vector figures; SVG retains editable text. For a two-column
paper, include the PDF in a figure* environment at width=\\textwidth.
Existing outputs are protected unless --overwrite is supplied.

This is a schematic, not a simulation or an image-generation API request.
All text, positions, colors, and connections are specified in this source.
It does not load checkpoints, run physics, or change benchmark results.

Basis:
  /home/aks-lab/Downloads/ARS_methodology-1.pdf
  /home/aks-lab/Downloads/Bimanual_AAMAS_2027-9.pdf, Sections 3.3-3.4
  manipulator_mujoco, revision 00898ab:
    real_demo/real_demo/ARS.py: ARSV2._rollout_single and ARSV2.train

Interpretation:
* The policy supplies state-dependent cost weights, NOT robot controls.
  Policy output conversion and phase-dependent costs are abstracted here.
* MPPI is deliberately one opaque block; MJX rollout internals are omitted.
* The applied control changes the MJX environment state. State feedback goes
  to BOTH the policy and MPPI at the next closed-loop step.
* Dashed connections are training only. ARS perturbs policy parameters,
  evaluates full closed-loop episodes, and updates parameters using returns.
  The return arrow originates from the closed-loop enclosure, not just MJX:
  rewards may also include planner diagnostics and phase/terminal events.
* During final inference ARS updates are absent; the trained policy is fixed.
* MJX here denotes the bimanual pipeline, not the quadrotor's CPU MuJoCo
  execution with analytical planning rollouts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


STEM = "ars_mppi_mjx_methodology_wide"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "methodology_figures"
CANVAS = (14.4, 4.85)
INK = "#20384B"
TRAINING = "#966124"
BACKGROUND = "#FAFBFD"

# Edit these labels and sizes to adapt the publication figure.
LABELS = {
    "policy": "Cost-weight\npolicy",
    "planner": "MPPI",
    "environment": "MJX\nenvironment",
    "training": "ARS training",
    "training_detail": "Perturb and update policy",
    "training_only": "Training only",
    "parameters": "Policy parameters",
    "returns": "Episode returns",
    "online": "Online control — repeated each step",
    "weights": "Cost weights",
    "action": "Control action",
    "state": "State feedback",
}
FONT = {"node": 22, "edge": 18.5, "heading": 18, "detail": 17.5}


@dataclass(frozen=True)
class Block:
    key: str
    x: float
    y: float
    width: float
    height: float
    fill: str

    @property
    def cx(self):
        return self.x + self.width / 2

    @property
    def cy(self):
        return self.y + self.height / 2

    def port(self, side):
        return {
            "left": (self.x, self.cy),
            "right": (self.x + self.width, self.cy),
            "top": (self.cx, self.y + self.height),
            "bottom": (self.cx, self.y),
        }[side]


BLOCKS = {
    "policy": Block("policy", 0.55, 1.35, 3.25, 1.02, "#E6EFF8"),
    "planner": Block("planner", 6.05, 1.35, 2.80, 1.02, "#EDF0FA"),
    "environment": Block("environment", 11.05, 1.35, 2.85, 1.02, "#E3F1ED"),
    "training": Block("training", 5.65, 3.30, 3.60, 1.00, "#FAEEDD"),
}


def add_label(ax, key, x, y, *, size="edge", color=INK, weight="normal", **kwargs):
    artist = ax.text(
        x, y, LABELS[key], ha="center", va="center",
        fontsize=FONT[size], color=color, fontweight=weight, zorder=5,
        **kwargs,
    )
    artist.set_gid("label-" + key)
    return artist


def add_arrow(ax, key, points, *, training=False):
    """Connect explicit ports with an editable orthogonal vector path."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    arrow = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color=TRAINING if training else INK,
        linestyle=(0, (5, 3)) if training else "-",
        capstyle="round", joinstyle="round", zorder=3,
    )
    arrow.set_gid("arrow-" + key)
    ax.add_patch(arrow)
    return arrow


def build_figure():
    """Return the figure so it can also be imported and edited in a notebook."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",  # Keep SVG labels as editable text, not paths.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })
    fig = plt.figure(figsize=CANVAS, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, CANVAS[0]), ylim=(0, CANVAS[1]))
    ax.axis("off")

    enclosure = FancyBboxPatch(
        (0.17, 0.10), 14.06, 2.85,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.2, edgecolor="#C7D2DC", facecolor=BACKGROUND, zorder=0,
    )
    enclosure.set_gid("online-control-enclosure")
    ax.add_patch(enclosure)
    add_label(ax, "online", 7.45, 2.73, size="heading", weight="semibold")
    add_label(ax, "training_only", 7.45, 4.61, size="heading", color=TRAINING)

    for key, block in BLOCKS.items():
        training = key == "training"
        patch = FancyBboxPatch(
            (block.x, block.y), block.width, block.height,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=block.fill, edgecolor=TRAINING if training else INK,
            linewidth=1.6, linestyle=(0, (5, 3)) if training else "-", zorder=2,
        )
        patch.set_gid("block-" + key)
        ax.add_patch(patch)
        add_label(
            ax, key, block.cx, block.cy + (0.18 if training else 0),
            size="node", weight="semibold", color=TRAINING if training else INK,
        )
        if training:
            add_label(ax, "training_detail", block.cx, block.cy - 0.22,
                      size="detail", color=TRAINING)

    policy, planner, environment, training = (
        BLOCKS[key] for key in ("policy", "planner", "environment", "training")
    )

    # Solid online-control chain: the policy sets costs, MPPI sets controls.
    add_arrow(ax, "cost-weights", [policy.port("right"), planner.port("left")])
    add_label(ax, "weights", 4.925, 2.09)
    add_arrow(ax, "control-action", [planner.port("right"), environment.port("left")])
    add_label(ax, "action", 9.95, 2.09)

    # The environment state feeds BOTH the weight policy and the MPPI solve.
    feedback_y = 0.63
    add_arrow(ax, "state-to-policy", [
        environment.port("bottom"), (environment.cx, feedback_y),
        (policy.cx, feedback_y), policy.port("bottom"),
    ])
    add_arrow(ax, "state-to-planner", [
        (planner.cx, feedback_y), planner.port("bottom"),
    ])
    ax.plot(planner.cx, feedback_y, "o", color=INK, markersize=4, zorder=4)
    add_label(ax, "state", 7.45, 0.30)

    # Dashed training-only connections. The return belongs to the whole episode.
    add_arrow(ax, "policy-parameters", [
        training.port("left"), (policy.cx, training.cy), policy.port("top"),
    ], training=True)
    add_label(ax, "parameters", 3.78, 4.08, color=TRAINING)
    add_arrow(ax, "episode-returns", [
        (12.475, 2.95), (12.475, training.cy), training.port("right"),
    ], training=True)
    add_label(ax, "returns", 10.86, 4.08, color=TRAINING)
    return fig


def validate_labels(fig):
    """Fail early if a source edit causes labels to overlap or get clipped."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = fig.axes[0].texts
    boxes = [(text, text.get_window_extent(renderer)) for text in texts]
    canvas = fig.bbox
    for text, box in boxes:
        if not (canvas.x0 <= box.x0 < box.x1 <= canvas.x1
                and canvas.y0 <= box.y0 < box.y1 <= canvas.y1):
            raise ValueError(f"Label outside canvas: {text.get_text()!r}")
    for index, (left, box) in enumerate(boxes):
        for right, other in boxes[index + 1:]:
            if box.overlaps(other):
                raise ValueError(f"Overlapping labels: {left.get_text()!r}, {right.get_text()!r}")
    # Require padding inside each block, not just a label inside the canvas.
    ax = fig.axes[0]
    for text, box in boxes:
        key = text.get_gid().removeprefix("label-")
        block = BLOCKS.get("training" if key == "training_detail" else key)
        if block is None:
            continue
        low = ax.transData.transform((block.x + 0.08, block.y + 0.08))
        high = ax.transData.transform((block.x + block.width - 0.08,
                                       block.y + block.height - 0.08))
        if not (low[0] <= box.x0 < box.x1 <= high[0]
                and low[1] <= box.y0 < box.y1 <= high[1]):
            raise ValueError(f"Label does not fit its block: {text.get_text()!r}")


def save_figure(output_dir=DEFAULT_OUTPUT_DIR, *, dpi=300, overwrite=False):
    output_dir = Path(output_dir)
    outputs = [output_dir / f"{STEM}.{ext}" for ext in ("pdf", "svg", "png")]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Use --overwrite to replace existing outputs: "
                              + ", ".join(map(str, existing)))
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    fig = build_figure()
    try:
        validate_labels(fig)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    parser.add_argument("--overwrite", action="store_true", help="Replace this figure's outputs")
    args = parser.parse_args()
    try:
        paths = save_figure(args.output_dir, dpi=args.dpi, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
