"""Plots for token-axis spectra.

Every function returns a Matplotlib figure and writes nothing. Saving, naming
and directory layout are the caller's business.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .spectrum import Spectrum


def _axes(ax=None, figsize=(9, 5)):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _axis_values(spectrum: Spectrum, x: str):
    if x == "period":
        return spectrum.periods, "Period (tokens/cycle)"
    if x == "frequency":
        return spectrum.freqs, "Frequency (cycles/token)"
    raise ValueError(f"x must be 'period' or 'frequency', not {x!r}")


def plot_spectrum(
    spectrum: Spectrum,
    *,
    x: str = "period",
    dims: Optional[Sequence[int]] = None,
    mark_period: Optional[float] = None,
    max_period: float = 32.0,
    ax=None,
    title: Optional[str] = None,
):
    """Dimension-averaged power, optionally with individual dimensions behind it.

    The period axis is the default because it is the one that can be checked
    against the text: a peak at 7 means something repeats every 7 tokens.

    `max_period` trims the long-period end of the plot, which is display only.
    That end of the axis is where a few bins stretch over a huge period range,
    so it eats most of the width while saying very little.
    """
    fig, ax = _axes(ax)
    axis, label = _axis_values(spectrum, x)

    keep = np.isfinite(axis)
    if x == "period":
        keep &= axis <= max_period
    order = np.argsort(axis[keep])
    xs = axis[keep][order]

    if dims:
        lookup = {int(d): i for i, d in enumerate(spectrum.dims)}
        for d in dims:
            if int(d) not in lookup:
                continue
            ys = spectrum.power[lookup[int(d)]][keep][order]
            ax.plot(xs, ys, linewidth=0.8, alpha=0.35, label=f"dim {d}")

    mean = spectrum.mean_power()[keep][order]
    ax.plot(xs, mean, color="#1f77b4", linewidth=2.0, label="mean over dimensions")

    if mark_period is not None:
        ax.axvline(mark_period, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"period {mark_period:g}")

    ax.set_xlabel(label)
    ax.set_ylabel("Power")
    if title:
        ax.set_title(title)
    elif spectrum.layer is not None:
        ax.set_title(f"Layer {spectrum.layer}, {spectrum.n_tokens} tokens")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_layer_sweep(
    spectra: Sequence[Spectrum],
    *,
    x: str = "period",
    mark_period: Optional[float] = None,
    max_period: float = 32.0,
    normalize: bool = True,
    ax=None,
    title: Optional[str] = None,
):
    """One curve per layer, for watching structure appear or fade with depth.

    Power grows by orders of magnitude through the network, so curves are scaled
    to their own maximum by default. Without that the deep layers are the only
    ones visible and the plot says nothing the y-axis did not already say.
    """
    fig, ax = _axes(ax, figsize=(9, 5.5))
    if not spectra:
        raise ValueError("nothing to plot")

    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(spectra)))
    for spec, color in zip(spectra, colors):
        axis, label = _axis_values(spec, x)
        keep = np.isfinite(axis)
        if x == "period":
            keep &= axis <= max_period
        order = np.argsort(axis[keep])
        ys = spec.mean_power()[keep][order]
        if normalize and ys.max() > 0:
            ys = ys / ys.max()
        ax.plot(axis[keep][order], ys, color=color, linewidth=1.4,
                label=f"layer {spec.layer}" if spec.layer is not None else None)

    if mark_period is not None:
        ax.axvline(mark_period, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"period {mark_period:g}")

    ax.set_xlabel(label)
    ax.set_ylabel("Power (scaled per layer)" if normalize else "Power")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_band_pair(spectrum: Spectrum, pair, *, ax=None, title: Optional[str] = None):
    """Where an equal-power band split falls on the spectrum.

    Takes a `BandPair` from `energy.equal_power_split`. The two shaded regions
    hold the same share of the total power, so their difference in width is the
    thing to look at.
    """
    fig, ax = _axes(ax)
    ax.semilogy(spectrum.freqs[1:], np.maximum(spectrum.mean_power()[1:], 1e-12),
                color="#333333", linewidth=1.4)
    ax.axvspan(*pair.low, color="#1f77b4", alpha=0.25,
               label=f"low band, {pair.low_power:.0%} of power ({pair.low_bins} bins)")
    ax.axvspan(*pair.high, color="#d62728", alpha=0.25,
               label=f"high band, {pair.high_power:.0%} of power ({pair.high_bins} bins)")
    ax.set_xlabel("Frequency (cycles/token)")
    ax.set_ylabel("Power")
    ax.set_title(title or f"Equal-power split at {pair.fraction:.0%}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
