"""Figure styling and the drawing primitives every figure needs."""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

# Colour-universal defaults. Named here so a change reaches every figure.
COLOR_UP = "#ff4b00"
COLOR_DOWN = "#005aff"
COLOR_SAME = "#bdbdbd"


def apply_style(style=None, rcparams=None):
    """The lab style if it is importable, then any explicit overrides.

    Vector text is forced on: a figure whose labels are outlines cannot be
    edited afterwards, and that is discovered at the point of submission.
    """
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    if style:
        plt.style.use(style)
    if rcparams:
        with open(rcparams) as fh:
            plt.rcParams.update(json.load(fh))


def roi_edge(entry, shape, offset=(0, 0)):
    """Boundary pixels of one ROI, shifted if the images were cropped."""
    m = np.zeros(shape, bool)
    yy = np.asarray(entry["ypix"]) - offset[0]
    xx = np.asarray(entry["xpix"]) - offset[1]
    k = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    m[yy[k], xx[k]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m & ~inner


def show_image(ax, img, clip=(0.5, 99.8), cmap="gray", title=None):
    if img is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center",
                transform=ax.transAxes, color="0.5", fontsize=9)
    else:
        img = np.asarray(img, float)
        lo, hi = np.percentile(img, list(clip))
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def scale_bar_um(ax, shape, pixel_size_um, length_um=50.0, color="w"):
    if not pixel_size_um:
        return
    bar = length_um / pixel_size_um
    x0, y0 = shape[1] * 0.06, shape[0] * 0.94
    ax.plot([x0, x0 + bar], [y0, y0], "-", color=color, lw=2.5,
            solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 - 3, f"{length_um:g} um", color=color,
            ha="center", va="bottom", fontsize=7)


def pad_for_markers(lo, hi, point_size, axes_width_in, min_frac=0.035):
    """Axis limits that leave room for a marker sitting on the limit.

    Marker size is in points and the data are not, so a point at zero is drawn
    as a half circle unless the padding is worked out from the figure size.
    """
    frac = (np.sqrt(point_size) / 2 + 2.0) / (axes_width_in * 72)
    pad = max(hi - lo, 1e-9) * max(frac, min_frac)
    return [lo - pad, hi + pad]


def save(fig, stem, dpi=200):
    """Always both: the PNG to look at, the PDF to put in a figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")
