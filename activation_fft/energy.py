"""Making band comparisons fair.

The obvious experiment — delete the low band, delete the high band, see which
hurts more — is rigged. Activation spectra fall off steeply with frequency, so
the bottom few bins hold most of the energy. Deleting them removes a large
fraction of the signal; deleting an equally wide slice at the top removes very
little. The low band wins that comparison whatever it encodes. At the limit:
drop the DC bin and the representation is gone.

So a band comparison only says something about *content* once the two edits are
matched on *magnitude*. Two ways to do that live here. `equal_power_split`
chooses the cutoffs so each band carries the same share of the total power, for
edits that delete a band outright. `match_noise_std` rescales the noise level so
two noise-type edits perturb the hidden state by the same relative energy, for
edits that are tuned by amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .perturb import apply_spectral_edit
from .spectrum import Spectrum


@dataclass
class BandPair:
    """A low band and a high band carrying a matched share of total power."""

    low: Tuple[float, float]
    high: Tuple[float, float]
    fraction: float
    low_power: float
    high_power: float
    low_bins: int
    high_bins: int

    @property
    def width_ratio(self) -> float:
        """How many times wider the high band is, to carry the same power."""
        low_width = self.low[1] - self.low[0]
        if low_width <= 0:
            return float("inf")
        return (self.high[1] - self.high[0]) / low_width

    def __str__(self) -> str:
        return (
            f"low {self.low[0]:.4f}-{self.low[1]:.4f} cyc/tok "
            f"({self.low_bins} bins, {self.low_power:.1%} of power) vs "
            f"high {self.high[0]:.4f}-{self.high[1]:.4f} cyc/tok "
            f"({self.high_bins} bins, {self.high_power:.1%}); "
            f"high band is {self.width_ratio:.1f}x wider"
        )


def power_profile(spectrum: Spectrum, skip_dc: bool = True) -> np.ndarray:
    """Power per bin, normalised to sum to one.

    DC is dropped by default. A detrended spectrum has almost none left in it,
    and leaving it in would let a residual mean dominate the shares.
    """
    power = spectrum.mean_power().astype(np.float64).copy()
    if skip_dc and power.size:
        power[0] = 0.0
    total = power.sum()
    if total <= 0:
        raise ValueError("spectrum carries no power outside DC")
    return power / total


def equal_power_split(
    spectrum: Spectrum,
    fraction: float = 0.2,
    *,
    tolerance: float = 0.5,
    min_gap_bins: int = 1,
) -> BandPair:
    """Cutoffs for a low and a high band that each hold `fraction` of the power.

    Deleting either band then removes the same amount of energy, so a difference
    in behavioural damage is a difference in what the bands carried.

    The two bands come out at very different widths, since matching on power
    forces the high band wider. `BandPair.width_ratio` is that factor.

    Bins are indivisible, so the request cannot always be met. A steep spectrum
    can put more than `fraction` of its power in the single lowest bin, and then
    no low band is small enough. That raises rather than returning a band which
    holds several times what was asked for; `tolerance` is how much overshoot
    counts as close enough.
    """
    if not 0 < fraction < 0.5:
        raise ValueError("fraction must be in (0, 0.5) so the two bands cannot overlap")

    share = power_profile(spectrum)
    freqs = spectrum.freqs
    ceiling = fraction * (1.0 + tolerance)

    lo_idx = int(np.searchsorted(np.cumsum(share), fraction))
    lo_idx = min(max(lo_idx, 1), share.size - 1)
    low_power = float(share[: lo_idx + 1].sum())

    hi_back = int(np.searchsorted(np.cumsum(share[::-1]), fraction))
    hi_idx = share.size - 1 - min(hi_back, share.size - 2)
    high_power = float(share[hi_idx:].sum())

    if low_power > ceiling:
        raise ValueError(
            f"cannot build a low band holding {fraction:.1%} of the power: the lowest "
            f"{lo_idx} bin(s) already hold {low_power:.1%}, and bins cannot be split. "
            f"Bin spacing is {spectrum.resolution:.4f} cycles/token at {spectrum.n_tokens} "
            f"tokens, so either analyse a longer sequence or ask for {low_power:.0%}."
        )
    if high_power > ceiling:
        raise ValueError(
            f"cannot build a high band holding {fraction:.1%} of the power without "
            f"overshooting to {high_power:.1%}; the spectrum is too concentrated at low "
            "frequency for this fraction"
        )
    if hi_idx - lo_idx < min_gap_bins + 1:
        raise ValueError(
            f"a low and a high band each holding {fraction:.1%} of the power leave no gap "
            f"between them (bins {lo_idx} and {hi_idx} of {share.size}); "
            "the spectrum is too concentrated, so use a smaller fraction"
        )

    return BandPair(
        low=(0.0, float(freqs[lo_idx])),
        high=(float(freqs[hi_idx]), float(freqs[-1])),
        fraction=fraction,
        low_power=low_power,
        high_power=high_power,
        low_bins=lo_idx + 1,
        high_bins=int(share.size - hi_idx),
    )


def edit_energy_ratio(
    hidden: torch.Tensor,
    op: str,
    *,
    n_tokens: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    **edit_kwargs,
) -> float:
    """Relative energy an edit puts into the hidden states.

    Mean squared change divided by mean squared original, so the number is
    comparable across layers and models.
    """
    with torch.no_grad():
        edited = apply_spectral_edit(
            hidden, op, n_tokens=n_tokens, generator=generator, **edit_kwargs
        )
        delta = (edited.float() - hidden.float()).pow(2).mean()
        base = hidden.float().pow(2).mean().clamp_min(1e-12)
    return float((delta / base).item())


def match_noise_std(
    hidden: torch.Tensor,
    reference: dict,
    candidate: dict,
    *,
    n_tokens: Optional[int] = None,
    bounds: Tuple[float, float] = (1e-4, 10.0),
    generator: Optional[torch.Generator] = None,
) -> float:
    """Noise level at which `candidate` perturbs as hard as `reference`.

    Both dicts are keyword sets for `apply_spectral_edit`, each including its own
    `op` and `noise_std`. Perturbation energy from additive noise grows with the
    square of the noise level, so one measurement of each is enough and the
    answer is a closed-form rescale rather than a search.

    This only applies to edits whose strength is set by an amplitude. An edit
    that deletes a band has a fixed energy cost that no noise level can change —
    use `equal_power_split` for those.
    """
    ref_op = dict(reference)
    cand_op = dict(candidate)
    ref_name = ref_op.pop("op")
    cand_name = cand_op.pop("op")
    base_std = float(cand_op.get("noise_std", 1.0))

    ref_energy = edit_energy_ratio(
        hidden, ref_name, n_tokens=n_tokens, generator=generator, **ref_op
    )
    cand_energy = edit_energy_ratio(
        hidden, cand_name, n_tokens=n_tokens, generator=generator, **cand_op
    )
    if ref_energy <= 0 or cand_energy <= 0:
        return base_std

    scaled = base_std * (ref_energy / cand_energy) ** 0.5
    return float(min(max(scaled, bounds[0]), bounds[1]))
