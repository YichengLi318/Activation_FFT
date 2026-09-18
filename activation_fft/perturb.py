"""Editing a layer's spectrum during the forward pass.

The analysis half of this package only looks. This half intervenes: transform a
layer's hidden states along the token axis, change selected frequency bands,
transform back, and let the rest of the network run on the result. If the model's
behaviour changes, whatever the band carried mattered.

Every edit goes through `apply_spectral_edit`, and the round trip is applied even
when nothing is edited. rFFT followed by irFFT is not exactly the identity in
floating point, so a run with `op="none"` measures how much of an effect belongs
to the round trip rather than to the edit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

FREQUENCY_OPS = ("none", "noise", "lowpass", "highpass", "bandstop", "band_noise", "band_attenuate")
ACTIVATION_OPS = ("act_noise", "act_zero")
OPS = FREQUENCY_OPS + ACTIVATION_OPS


def decoder_layers(model) -> torch.nn.ModuleList:
    """The list of transformer blocks, for the two layouts in common use."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        f"cannot find decoder layers on {type(model).__name__}; "
        "expected model.model.layers or model.transformer.h"
    )


def resolve_layer_index(model, layer: int) -> int:
    layers = decoder_layers(model)
    idx = layer + len(layers) if layer < 0 else layer
    if not 0 <= idx < len(layers):
        raise IndexError(f"layer {layer} out of range for a {len(layers)}-layer model")
    return idx


def apply_spectral_edit(
    hidden: torch.Tensor,
    op: str = "none",
    *,
    dims: Optional[Sequence[int]] = None,
    noise_std: float = 1.0,
    low_cutoff: Optional[float] = None,
    high_cutoff: Optional[float] = None,
    band_gain: Optional[float] = None,
    n_tokens: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Edit `hidden` (batch, tokens, dim) along its token axis.

    Cutoffs are in cycles per token and so live between 0 and 0.5.

    `n_tokens` restricts the transform to the first that many positions and
    leaves the rest untouched, which is what you want whenever the batch is
    padded. Without it the pad tail is transformed along with the text and
    contributes energy to the low bins.
    """
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; choose from {list(OPS)}")
    if hidden.dim() != 3:
        raise ValueError(f"expected (batch, tokens, dim), got shape {tuple(hidden.shape)}")

    seq_len = hidden.shape[1]
    span = seq_len if n_tokens is None else min(int(n_tokens), seq_len)
    if span < 2:
        return hidden

    if span < seq_len:
        head = apply_spectral_edit(
            hidden[:, :span, :], op, dims=dims, noise_std=noise_std,
            low_cutoff=low_cutoff, high_cutoff=high_cutoff, band_gain=band_gain,
            generator=generator,
        )
        return torch.cat([head, hidden[:, span:, :]], dim=1)

    hidden_dim = hidden.shape[2]
    transposed = hidden.transpose(1, 2)  # (batch, dim, tokens)

    mask = None
    target = transposed
    if dims:
        valid = [d for d in dims if 0 <= d < hidden_dim]
        if not valid:
            raise ValueError(f"none of dims={list(dims)} are valid for {hidden_dim} hidden dimensions")
        mask = torch.zeros(hidden_dim, device=hidden.device, dtype=torch.bool)
        mask[valid] = True
        target = transposed[:, mask, :]

    def _restore(edited: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return edited.transpose(1, 2)
        out = transposed.clone()
        out[:, mask, :] = edited
        return out.transpose(1, 2)

    def _randn(like: torch.Tensor) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(like)
        return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)

    if op == "act_noise":
        scale = target.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return _restore(target + _randn(target) * noise_std * scale)
    if op == "act_zero":
        return _restore(torch.zeros_like(target))

    # cuFFT refuses half precision at non-power-of-two lengths, and token counts
    # are rarely powers of two.
    orig_dtype = target.dtype
    work = target.float() if orig_dtype in (torch.float16, torch.bfloat16) else target

    spec = torch.fft.rfft(work, dim=-1)
    freqs = torch.fft.rfftfreq(work.shape[-1], d=1.0, device=hidden.device)

    if op == "noise":
        scale = spec.abs().mean().clamp_min(1e-6)
        spec = spec + torch.complex(_randn(spec.real), _randn(spec.imag)) * noise_std * scale
    elif op == "lowpass":
        _require(op, low_cutoff=low_cutoff)
        spec = spec * (freqs <= low_cutoff).view(1, 1, -1)
    elif op == "highpass":
        _require(op, high_cutoff=high_cutoff)
        spec = spec * (freqs >= high_cutoff).view(1, 1, -1)
    elif op == "bandstop":
        _require(op, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        spec = spec * ((freqs < low_cutoff) | (freqs > high_cutoff)).view(1, 1, -1)
    elif op == "band_attenuate":
        _require(op, low_cutoff=low_cutoff, high_cutoff=high_cutoff, band_gain=band_gain)
        band = ((freqs >= low_cutoff) & (freqs <= high_cutoff)).view(1, 1, -1).to(spec.real.dtype)
        spec = spec * (1.0 - band + band * band_gain)
    elif op == "band_noise":
        _require(op, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        band = ((freqs >= low_cutoff) & (freqs <= high_cutoff)).view(1, 1, -1).to(spec.real.dtype)
        denom = band.sum().clamp_min(1.0)
        band_mean = (spec * band).sum(dim=-1, keepdim=True) / denom
        # Mean absolute deviation inside the band, used to scale the noise so the
        # replacement sits at the same order of magnitude as what it replaces.
        spread = ((spec - band_mean).abs() * band).sum(dim=-1, keepdim=True) / denom
        spread = spread.clamp_min(1e-6)
        noise = torch.complex(_randn(spec.real), _randn(spec.imag)) * noise_std * spread + band_mean
        spec = spec * (1.0 - band) + noise * band

    edited = torch.fft.irfft(spec, n=work.shape[-1], dim=-1)
    if edited.dtype != orig_dtype:
        edited = edited.to(orig_dtype)
    return _restore(edited)


def _require(op: str, **kwargs) -> None:
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"op {op!r} needs {', '.join(missing)}")


@dataclass
class SpectralEdit:
    """A forward hook that edits one layer's spectrum in place."""

    op: str = "none"
    dims: Optional[Sequence[int]] = None
    noise_std: float = 1.0
    low_cutoff: Optional[float] = None
    high_cutoff: Optional[float] = None
    band_gain: Optional[float] = None
    n_tokens: Optional[int] = None
    generator: Optional[torch.Generator] = field(default=None, repr=False)
    calls: int = field(default=0, init=False)

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            edited = apply_spectral_edit(
                hidden,
                self.op,
                dims=self.dims,
                noise_std=self.noise_std,
                low_cutoff=self.low_cutoff,
                high_cutoff=self.high_cutoff,
                band_gain=self.band_gain,
                n_tokens=self.n_tokens,
                generator=self.generator,
            )
        self.calls += 1
        if isinstance(output, tuple):
            return (edited,) + tuple(output[1:])
        return edited


@contextmanager
def spectral_edit(model, layer: int, edit: SpectralEdit):
    """Attach `edit` to one layer for the duration of the block."""
    idx = resolve_layer_index(model, layer)
    handle = decoder_layers(model)[idx].register_forward_hook(edit)
    try:
        yield idx
    finally:
        handle.remove()


def split_bands(n_bands: int, f_max: float = 0.5) -> List[Tuple[float, float]]:
    """`n_bands` equal-width bands covering 0 to `f_max` cycles per token.

    Equal width, not equal power. Low bands carry far more energy than high
    ones, so damage from removing an equal-width band says as much about how
    much energy went missing as about what the band encoded. See `energy` for
    the equal-power alternative.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be at least 1")
    edges = [f_max * i / n_bands for i in range(n_bands + 1)]
    return list(zip(edges[:-1], edges[1:]))
