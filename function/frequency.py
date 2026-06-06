from typing import Dict, List, Optional, Tuple

from function.perturb import FrequencyHook, _run_single_text


def run_band_noise_generations(
    model,
    tokenizer,
    device: str,
    prompt: str,
    layer_index: int,
    max_length: int,
    max_new_tokens: int,
    noise_std: float,
    max_time: Optional[float] = None,
    topk: int = 1,
    progress_desc: Optional[str] = None,
    return_attentions: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    filter_outputs: Dict[str, Dict[str, float]] = {}
    filter_keys: List[str] = []
    max_freq = 0.5
    iterator = range(10)
    if progress_desc:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=progress_desc, leave=False)
        except Exception:
            iterator = range(10)
    for i in iterator:
        high = max_freq * (1.0 - 0.1 * i)
        low = max_freq * (1.0 - 0.1 * (i + 1))
        label = f"band_{int((1.0 - 0.1*(i+1))*100)}_{int((1.0 - 0.1*i)*100)}"
        hook = FrequencyHook(
            op_type="band_noise",
            dims=None,
            noise_std=noise_std,
            low_cutoff=low,
            high_cutoff=high,
            force_roundtrip=True,
        )
        pert = _run_single_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=prompt,
            max_length=max_length,
            layer_index=layer_index,
            hook=hook,
            topk=topk,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_time=max_time,
            return_attentions=return_attentions,
        )
        filter_outputs[label] = {
            "text": (pert.get("generated") or "").strip(),
            "band_low": low,
            "band_high": high,
            "attentions": pert.get("attentions"),
            "attn_valid_len": pert.get("attn_valid_len"),
        }
        filter_keys.append(label)
    return filter_outputs, filter_keys
