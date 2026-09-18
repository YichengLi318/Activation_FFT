"""Pulling hidden states out of a local causal language model.

Any model `transformers` can load with AutoModelForCausalLM works here, given a
Hub id or a path on disk. There is no assumption about where the weights live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: str
    name: str = ""

    @property
    def n_layers(self) -> int:
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            n = getattr(cfg, attr, None)
            if n:
                return int(n)
        raise AttributeError(f"cannot find a layer count on {type(cfg).__name__}")

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        for attr in ("hidden_size", "n_embd", "d_model"):
            n = getattr(cfg, attr, None)
            if n:
                return int(n)
        raise AttributeError(f"cannot find a hidden size on {type(cfg).__name__}")


@dataclass
class Activations:
    """Hidden states for one text at one layer, padding already removed."""

    values: np.ndarray  # (n_tokens, hidden_size)
    tokens: List[str]
    layer: int

    @property
    def n_tokens(self) -> int:
        return self.values.shape[0]


def load_model(
    name_or_path: str,
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a causal LM and its tokenizer.

    `name_or_path` goes to `transformers` unchanged, so a Hub id, a relative
    path and an absolute path all behave the way you would expect.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return LoadedModel(model=model, tokenizer=tokenizer, device=device, name=str(name_or_path))


def from_model(model, tokenizer, device: Optional[str] = None, name: str = "") -> LoadedModel:
    """Wrap a model you already hold, instead of loading one from disk."""
    if device is None:
        device = str(next(model.parameters()).device)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device, name=name)


def resolve_layers(lm: LoadedModel, layers: Optional[Sequence[int]]) -> List[int]:
    """Normalise a layer selection against the model depth.

    Layer 0 is the output of the first transformer block and layer n-1 the
    output of the last, matching `hidden_states[1:]`. The embedding output is
    not addressable; it carries no contextual mixing and its spectrum is a
    property of the token sequence rather than of the model.
    """
    n = lm.n_layers
    if layers is None:
        return list(range(n))
    out = []
    for layer in layers:
        idx = layer + n if layer < 0 else layer
        if not 0 <= idx < n:
            raise IndexError(f"layer {layer} out of range for a {n}-layer model")
        out.append(idx)
    return out


def count_tokens(lm: LoadedModel, texts: Union[str, Iterable[str]]) -> List[int]:
    """Token count per text, with no padding or truncation.

    Worth running before any spectral analysis that assumes a period: the
    tokenizer, not the writing system, decides how many positions a line takes.
    """
    if isinstance(texts, str):
        texts = [texts]
    return [len(lm.tokenizer(t, padding=False, truncation=False)["input_ids"]) for t in texts]


def hidden_states(
    lm: LoadedModel,
    texts: Union[str, Iterable[str]],
    *,
    layers: Optional[Sequence[int]] = None,
    max_length: Optional[int] = None,
) -> List[Dict[int, Activations]]:
    """Hidden states for each text, keyed by layer.

    Texts run one at a time and unpadded, so every returned array is exactly as
    long as its text. Batching would buy speed and cost correctness: padded
    positions carry hidden states, and a constant tail is indistinguishable from
    low-frequency signal once transformed.

    One forward pass yields every layer, so asking for a subset of `layers`
    saves memory but not compute.
    """
    if isinstance(texts, str):
        texts = [texts]
    wanted = resolve_layers(lm, layers)

    out: List[Dict[int, Activations]] = []
    for text in texts:
        encoded = lm.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        ids = encoded["input_ids"]
        encoded = {k: v.to(lm.device) for k, v in encoded.items() if hasattr(v, "to")}

        with torch.no_grad():
            states = lm.model(**encoded, output_hidden_states=True, use_cache=False).hidden_states

        tokens = lm.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        out.append(
            {
                layer: Activations(
                    values=states[layer + 1][0].float().cpu().numpy(),
                    tokens=tokens,
                    layer=layer,
                )
                for layer in wanted
            }
        )
    return out
