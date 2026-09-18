"""Discrete Fourier transform of hidden activations along the token axis.

Nothing here touches a model. The input is an array of activations; the output
is a Spectrum. So the transform can be checked against signals whose period is
known in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "boxcar": np.ones,
}


@dataclass
class Spectrum:
    """One-sided spectrum of a set of hidden dimensions over token position.

    Frequencies are in cycles per token, so the axis runs from 0 to 0.5. The
    reciprocal (tokens per cycle) is usually the more readable view and is
    available as `periods`.
    """

    freqs: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    dims: np.ndarray
    n_tokens: int
    layer: Optional[int] = None
    window: str = "hann"
    detrended: bool = True
    normalized: bool = True
    n_samples: int = 1
    amplitude_std: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def power(self) -> np.ndarray:
        return self.amplitude ** 2

    @property
    def periods(self) -> np.ndarray:
        """Tokens per cycle. The DC bin has no period and comes back as inf."""
        with np.errstate(divide="ignore"):
            return np.where(self.freqs > 0, 1.0 / np.where(self.freqs > 0, self.freqs, 1.0), np.inf)

    @property
    def resolution(self) -> float:
        """Spacing between neighbouring frequency bins, in cycles per token."""
        return 1.0 / self.n_tokens

    def mean_power(self) -> np.ndarray:
        """Power averaged over dimensions, shape (n_freq,)."""
        return self.power.mean(axis=0)

    def peak(self, min_period: float = 2.0, max_period: Optional[float] = None):
        """Strongest bin of the dimension-averaged power spectrum.

        Returns (period, frequency, power). The DC and near-DC bins are excluded
        by the `min_period` / `max_period` window rather than by a fixed offset;
        a hard bin offset hides whatever is actually going on down there.
        """
        p = self.periods
        keep = np.isfinite(p) & (p >= min_period)
        if max_period is not None:
            keep &= p <= max_period
        if not keep.any():
            raise ValueError(
                f"no bins with period in [{min_period}, {max_period}]; "
                f"sequence is {self.n_tokens} tokens, resolution {self.resolution:.4f} cycles/token"
            )
        power = self.mean_power()
        idx = np.flatnonzero(keep)[np.argmax(power[keep])]
        return float(p[idx]), float(self.freqs[idx]), float(power[idx])

    def power_at_period(self, period: float) -> float:
        """Power in the bin closest to `period` tokens per cycle."""
        target = 1.0 / period
        idx = int(np.argmin(np.abs(self.freqs - target)))
        return float(self.mean_power()[idx])

    def harmonics(self, period: float, n: int = 3):
        """Power at the fundamental and its first `n - 1` harmonics.

        A real periodicity usually shows up at 2f and 3f as well, so this is the
        cheapest check that a peak is not just one noisy bin.
        """
        out = []
        base = 1.0 / period
        for k in range(1, n + 1):
            f = base * k
            if f > self.freqs[-1]:
                break
            idx = int(np.argmin(np.abs(self.freqs - f)))
            out.append((k, float(self.freqs[idx]), float(self.mean_power()[idx])))
        return out


def min_tokens_to_separate(period_a: float, period_b: float, margin: float = 1.0) -> int:
    """Shortest sequence whose bin spacing can tell two periods apart.

    Bin spacing is 1/N cycles per token, so two periods land in different bins
    only when N > 1 / |1/a - 1/b|. Separating period 5 from period 7 therefore
    needs at least 18 tokens; `margin` asks for that many bins of separation
    instead of one.
    """
    gap = abs(1.0 / period_a - 1.0 / period_b)
    if gap == 0:
        raise ValueError("the two periods are equal")
    return int(np.ceil(margin / gap))


def token_spectrum(
    activations: ArrayLike,
    *,
    n_tokens: Optional[int] = None,
    dims: Optional[Sequence[int]] = None,
    window: str = "hann",
    detrend: bool = True,
    normalize_dims: bool = True,
    layer: Optional[int] = None,
) -> Spectrum:
    """Transform one sample of activations along its token axis.

    `activations` is (n_tokens, n_dim), the layout `hidden_states` returns.

    `n_tokens` truncates the sequence before the transform and exists for one
    reason: padded positions still carry hidden states, and a constant tail
    dumps energy into the low bins where it is easy to mistake for signal. Pass
    the real token count and the padding never enters the transform.
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_tokens, n_dim), got shape {x.shape}")

    if n_tokens is not None:
        if n_tokens < 2:
            raise ValueError(f"need at least 2 real tokens to transform, got {n_tokens}")
        x = x[:n_tokens]
    if x.shape[0] < 2:
        raise ValueError(f"need at least 2 tokens to transform, got {x.shape[0]}")

    if dims is None:
        dim_idx = np.arange(x.shape[1])
    else:
        dim_idx = np.asarray([d for d in dims if 0 <= d < x.shape[1]], dtype=int)
        if dim_idx.size == 0:
            raise ValueError(f"none of dims={list(dims)} are valid for {x.shape[1]} hidden dimensions")
    x = x[:, dim_idx].T  # (n_dim, n_tokens)

    if detrend:
        x = x - x.mean(axis=1, keepdims=True)
    if normalize_dims:
        # Hidden dimensions differ in scale by orders of magnitude. Without this
        # a handful of loud dimensions decide the dimension-averaged spectrum.
        scale = x.std(axis=1, keepdims=True)
        x = x / np.where(scale > 0, scale, 1.0)

    if window not in _WINDOWS:
        raise ValueError(f"unknown window {window!r}; choose from {sorted(_WINDOWS)}")
    n = x.shape[1]
    x = x * _WINDOWS[window](n)

    spec = np.fft.rfft(x, axis=1)
    return Spectrum(
        freqs=np.fft.rfftfreq(n, d=1.0),
        amplitude=np.abs(spec),
        phase=np.angle(spec),
        dims=dim_idx,
        n_tokens=n,
        layer=layer,
        window=window,
        detrended=detrend,
        normalized=normalize_dims,
    )


def average_spectra(spectra: Sequence[Spectrum]) -> Spectrum:
    """Average amplitude across samples that share a frequency axis.

    Samples of differing length have differing bin spacing, so mixing them would
    average unrelated frequencies together. That is refused rather than resampled.
    """
    if not spectra:
        raise ValueError("nothing to average")
    first = spectra[0]
    for s in spectra[1:]:
        if s.n_tokens != first.n_tokens:
            raise ValueError(
                f"cannot average spectra of different lengths ({first.n_tokens} vs {s.n_tokens}); "
                "transform a fixed token window, or average the samples separately"
            )
        if not np.array_equal(s.dims, first.dims):
            raise ValueError("cannot average spectra taken over different hidden dimensions")

    amps = np.stack([s.amplitude for s in spectra])
    return Spectrum(
        freqs=first.freqs,
        amplitude=amps.mean(axis=0),
        phase=np.stack([s.phase for s in spectra]).mean(axis=0),
        dims=first.dims,
        n_tokens=first.n_tokens,
        layer=first.layer,
        window=first.window,
        detrended=first.detrended,
        normalized=first.normalized,
        n_samples=len(spectra),
        amplitude_std=amps.std(axis=0),
    )
