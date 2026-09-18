"""Token-axis Fourier analysis of language-model activations.

Take the hidden states of one layer, read them as a signal over token position,
and transform. Peaks in the result are periodic patterns in what the layer is
representing, measured in tokens per cycle.

    import activation_fft as af

    lm = af.load_model("Qwen/Qwen3-1.7B")
    acts = af.hidden_states(lm, text, layers=[20])[0][20]
    spec = af.token_spectrum(acts.values, layer=20)
    print(spec.peak())

The same transform runs in reverse during a forward pass: edit a frequency band,
put the signal back, and see what the model does differently.

The README's design notes cover why any of this is a reasonable thing to do.
"""

from .energy import BandPair, edit_energy_ratio, equal_power_split, match_noise_std, power_profile
from .extract import (
    Activations,
    LoadedModel,
    count_tokens,
    from_model,
    hidden_states,
    load_model,
    resolve_layers,
)
from .perturb import (
    OPS,
    SpectralEdit,
    apply_spectral_edit,
    decoder_layers,
    resolve_layer_index,
    spectral_edit,
    split_bands,
)
from .plot import plot_band_pair, plot_layer_sweep, plot_spectrum
from .spectrum import Spectrum, average_spectra, min_tokens_to_separate, token_spectrum

__version__ = "0.1.0"

__all__ = [
    "Activations",
    "BandPair",
    "LoadedModel",
    "OPS",
    "SpectralEdit",
    "Spectrum",
    "apply_spectral_edit",
    "average_spectra",
    "count_tokens",
    "decoder_layers",
    "edit_energy_ratio",
    "equal_power_split",
    "from_model",
    "hidden_states",
    "load_model",
    "match_noise_std",
    "min_tokens_to_separate",
    "plot_band_pair",
    "plot_layer_sweep",
    "plot_spectrum",
    "power_profile",
    "resolve_layer_index",
    "resolve_layers",
    "spectral_edit",
    "split_bands",
    "token_spectrum",
]
