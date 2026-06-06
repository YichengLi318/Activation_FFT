import json
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from function import activation

# Keep thread usage low for consistency with activation.py
torch.set_num_threads(1)


def _get_decoder_layers(model):
    """
    Locate the list of decoder/transformer layers for common HF causal LM layouts.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Could not locate decoder layers on the model (expected model.model.layers or model.transformer.h).")


def _resolve_layer_index(layers, layer_index: int) -> int:
    """
    Normalize possibly-negative layer index against available layers.
    """
    if not layers:
        raise ValueError("No layers available on the model.")
    if layer_index < 0:
        layer_index = len(layers) + layer_index
    layer_index = max(0, min(layer_index, len(layers) - 1))
    return layer_index


def _apply_frequency_operation(
    hidden_states: torch.Tensor,
    op_type: str,
    dims: Optional[List[int]],
    noise_std: float,
    low_cutoff: Optional[float],
    high_cutoff: Optional[float],
    force_roundtrip: bool = True,
    band_gain: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply spectral edits on the sequence axis of hidden_states (batch, seq, hidden):
    - op_type: 'noise', 'lowpass', 'highpass', 'bandstop', or 'none'
    - dims: optional subset of hidden dims to edit; if None or empty, edit all
    - low_cutoff/high_cutoff: frequency cutoffs in cycles/token (0~0.5 typical)
    - force_roundtrip: when True, even 'none' will still rFFT+iFFT to mirror extraction+inversion
    """
    if hidden_states is None or hidden_states.dim() != 3:
        return hidden_states
    batch, seq_len, hidden = hidden_states.shape
    if seq_len < 2:
        # Not enough sequence length for a meaningful DFT
        return hidden_states

    transposed = hidden_states.transpose(1, 2)  # (batch, hidden, seq)

    # Select dimensions to edit
    dim_mask = None
    target = transposed
    if dims:
        valid_dims = [d for d in dims if 0 <= d < hidden]
        if not valid_dims:
            return hidden_states
        dim_mask = torch.zeros(hidden, device=hidden_states.device, dtype=torch.bool)
        dim_mask[valid_dims] = True
        target = transposed[:, dim_mask, :]

    if op_type == 'act_noise':
        # Direct activation-domain Gaussian noise on selected components.
        scale = target.std(dim=-1, keepdim=True).clamp_min(1e-6)
        noisy = target + torch.randn_like(target) * noise_std * scale
        if dim_mask is None:
            return noisy.transpose(1, 2)
        restored = transposed.clone()
        restored[:, dim_mask, :] = noisy
        return restored.transpose(1, 2)

    if op_type == 'act_zero':
        zeroed = torch.zeros_like(target)
        if dim_mask is None:
            return zeroed.transpose(1, 2)
        restored = transposed.clone()
        restored[:, dim_mask, :] = zeroed
        return restored.transpose(1, 2)

    # Run FFT in float32 to avoid half-precision cuFFT limits on non-power-of-two lengths.
    orig_dtype = target.dtype
    fft_input = target
    if fft_input.dtype in (torch.float16, torch.bfloat16):
        fft_input = fft_input.float()

    freq_repr = torch.fft.rfft(fft_input, dim=-1)
    freq_axis = torch.fft.rfftfreq(fft_input.shape[-1], d=1.0, device=hidden_states.device)

    if op_type == 'noise':
        base_scale = freq_repr.abs().mean().clamp_min(1e-6)
        noise_real = torch.randn_like(freq_repr.real) * noise_std * base_scale
        noise_imag = torch.randn_like(freq_repr.imag) * noise_std * base_scale
        freq_repr = freq_repr + torch.complex(noise_real, noise_imag)
    elif op_type == 'lowpass' and low_cutoff is not None:
        mask = (freq_axis <= low_cutoff).view(1, 1, -1)
        freq_repr = freq_repr * mask
    elif op_type == 'highpass' and high_cutoff is not None:
        mask = (freq_axis >= high_cutoff).view(1, 1, -1)
        freq_repr = freq_repr * mask
    elif op_type == 'bandstop' and low_cutoff is not None and high_cutoff is not None:
        mask = ((freq_axis < low_cutoff) | (freq_axis > high_cutoff)).view(1, 1, -1)
        freq_repr = freq_repr * mask
    elif op_type == 'band_attenuate' and low_cutoff is not None and high_cutoff is not None and band_gain is not None:
        band_mask = ((freq_axis >= low_cutoff) & (freq_axis <= high_cutoff)).view(1, 1, -1)
        band_mask = band_mask.to(freq_repr.dtype)
        freq_repr = freq_repr * (1.0 - band_mask + band_mask * band_gain)
    elif op_type == 'band_noise' and low_cutoff is not None and high_cutoff is not None:
        band_mask = ((freq_axis >= low_cutoff) & (freq_axis <= high_cutoff)).view(1, 1, -1)
        band_mask = band_mask.to(freq_repr.dtype)
        # keep band mean to preserve average component
        band_mask_f = band_mask.to(freq_repr.real.dtype)
        denom = band_mask_f.sum().clamp_min(1.0)
        band_mean = (freq_repr * band_mask_f).sum(dim=-1, keepdim=True) / denom
        band_var = ((freq_repr - band_mean).abs() * band_mask_f).sum(dim=-1, keepdim=True) / denom
        band_scale = band_var.clamp_min(1e-6)
        noise_real = torch.randn_like(freq_repr.real) * noise_std * band_scale
        noise_imag = torch.randn_like(freq_repr.imag) * noise_std * band_scale
        noise = torch.complex(noise_real, noise_imag) + band_mean
        freq_repr = freq_repr * (1.0 - band_mask) + noise * band_mask
    elif op_type == 'none' and not force_roundtrip:
        # No spectral edit and no forced round-trip: bypass
        return hidden_states
    # else: 'none' with force_roundtrip=True will still go through rFFT+iFFT

    modified = torch.fft.irfft(freq_repr, n=fft_input.shape[-1], dim=-1)

    if dim_mask is None:
        restored = modified.transpose(1, 2)
    else:
        restored = transposed.clone()
        restored[:, dim_mask, :] = modified
        restored = restored.transpose(1, 2)

    if restored.dtype != orig_dtype:
        restored = restored.to(orig_dtype)

    return restored


class FrequencyHook:
    """
    Forward hook that edits the frequency spectrum of hidden states on a target layer.
    """

    def __init__(
        self,
        op_type: str,
        dims: Optional[List[int]],
        noise_std: float,
        low_cutoff: Optional[float],
        high_cutoff: Optional[float],
        force_roundtrip: bool = True,
        band_gain: Optional[float] = None,
    ):
        self.op_type = op_type
        self.dims = dims or []
        self.noise_std = noise_std
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.force_roundtrip = force_roundtrip
        self.band_gain = band_gain

    def __call__(self, _module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            edited = _apply_frequency_operation(
                hidden_states=hidden,
                op_type=self.op_type,
                dims=self.dims,
                noise_std=self.noise_std,
                low_cutoff=self.low_cutoff,
                high_cutoff=self.high_cutoff,
                force_roundtrip=self.force_roundtrip,
                band_gain=self.band_gain,
            )
        if isinstance(output, tuple):
            as_list = list(output)
            as_list[0] = edited
            return tuple(as_list)
        return edited


@contextmanager
def register_layer_hook(model, layer_index: int, hook_fn):
    layers = _get_decoder_layers(model)
    resolved_idx = _resolve_layer_index(layers, layer_index)
    handle = layers[resolved_idx].register_forward_hook(hook_fn)
    try:
        yield resolved_idx
    finally:
        handle.remove()


def _topk_from_logits(logits: torch.Tensor, tokenizer, topk: int) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(probs, k=min(topk, probs.shape[-1]))
    result = []
    for p, idx in zip(top_values.tolist(), top_indices.tolist()):
        result.append({
            "token": tokenizer.decode([idx]),
            "token_id": int(idx),
            "prob": float(p),
        })
    return result


def _run_single_text(
    model,
    tokenizer,
    device: str,
    text: str,
    max_length: int,
    layer_index: int,
    hook: Optional[FrequencyHook],
    topk: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    return_log_probs: bool = False,
    return_attentions: bool = False,
    max_time: Optional[float] = None,
) -> Dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) if hasattr(v, 'to') else v for k, v in encoded.items()}

    ctx = register_layer_hook(model, layer_index, hook) if hook else nullcontext(layer_index)

    with torch.no_grad():
        with ctx as resolved_layer:
            outputs = model(
                **encoded,
                use_cache=False,
                output_attentions=return_attentions,
                return_dict=True,
            )
            logits = outputs.logits  # (1, seq, vocab)

            attn_mask = encoded.get('attention_mask')
            if attn_mask is not None:
                last_idx = int(attn_mask[0].sum().item()) - 1
            else:
                last_idx = encoded['input_ids'].shape[1] - 1
            last_idx = max(last_idx, 0)

            last_logits = logits[0, last_idx]
            top_tokens = _topk_from_logits(last_logits, tokenizer, topk)
            last_log_probs = None
            if return_log_probs:
                last_log_probs = torch.log_softmax(last_logits, dim=-1).detach().cpu()
            generated = None

            if max_new_tokens > 0:
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=getattr(tokenizer, 'eos_token_id', None),
                )
                if max_time is not None:
                    gen_kwargs["max_time"] = max_time
                if temperature is not None and temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["do_sample"] = True
                gen_ids = model.generate(
                    **encoded,
                    **gen_kwargs,
                )
                new_tokens = gen_ids[0, encoded['input_ids'].shape[1]:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

    result = {
        "layer_index": layer_index if hook is None else resolved_layer,
        "last_token_index": last_idx,
        "topk": top_tokens,
        "generated": generated,
        "seq_length": int(encoded['input_ids'].shape[1]),
        "last_log_probs": last_log_probs,
    }
    if return_attentions:
        attn = None
        try:
            if outputs.attentions:
                attn = outputs.attentions[-1]
        except Exception:
            attn = None
        result["attentions"] = attn
        result["attn_valid_len"] = int(last_idx + 1)
    return result


def run_frequency_perturbation(
    model,
    tokenizer,
    dataset_path: str,
    layer_index: int,
    operation: str,
    output_root: str,
    device: str,
    dims: Optional[List[int]] = None,
    noise_std: float = 0.05,
    low_cutoff: Optional[float] = None,
    high_cutoff: Optional[float] = None,
    topk: int = 5,
    max_new_tokens: int = 0,
    fast_mode: bool = False,
    sample_limit: Optional[int] = None,
):
    """
    For each text in the dataset:
    - run a baseline forward pass (no spectral edit) to capture logits/generation
    - run a perturbed forward pass where the chosen layer's hidden states are edited
      in the frequency domain, inverse-transformed, and fed forward
    - write both outcomes to output/perturbation/<dataset>/<op>_layerX/<category>/sample_*.json
    """
    corpus_name, items_by_category = activation._read_corpus_texts(dataset_path)

    all_texts = []
    for _cat, _txts in items_by_category.items():
        all_texts.extend(_txts)
    target_len = activation._compute_uniform_token_length(tokenizer, all_texts, fast_mode=fast_mode)

    op_tag = f"{operation}_layer{layer_index}"
    base_dir = os.path.join(output_root, corpus_name, op_tag)
    os.makedirs(base_dir, exist_ok=True)

    hook = FrequencyHook(
        op_type=operation,
        dims=dims,
        noise_std=noise_std,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )

    for category, texts in items_by_category.items():
        if fast_mode:
            texts = texts[:5]
        if sample_limit is not None:
            texts = texts[:sample_limit]

        out_dir = os.path.join(base_dir, category)
        os.makedirs(out_dir, exist_ok=True)

        for idx, text in enumerate(tqdm(texts, desc=f"Perturbing {category}")):
            baseline = _run_single_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                max_length=target_len,
                layer_index=layer_index,
                hook=None,
                topk=topk,
                max_new_tokens=max_new_tokens,
            )
            perturbed = _run_single_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                max_length=target_len,
                layer_index=layer_index,
                hook=hook,
                topk=topk,
                max_new_tokens=max_new_tokens,
            )

            payload = {
                "text_index": idx,
                "category": category,
                "dataset": corpus_name,
                "layer_index": perturbed.get("layer_index", layer_index),
                "operation": operation,
                "dims": dims or [],
                "noise_std": noise_std,
                "low_cutoff": low_cutoff,
                "high_cutoff": high_cutoff,
                "topk": topk,
                "max_new_tokens": max_new_tokens,
                "target_token_length": target_len,
                "original_text": text,
                "baseline": baseline,
                "perturbed": perturbed,
            }

            out_path = os.path.join(out_dir, f"sample_{idx}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Spectral perturbation outputs saved under: {base_dir}")
