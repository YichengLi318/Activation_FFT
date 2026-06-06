import json
import os
import random
import sys
import hashlib
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import matplotlib
import torch
from datasets import load_from_disk
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function import activation
from function.agreement_controls import (
    collect_calibration_vectors_and_labels,
    select_diffmean_dims_from_calibration,
    select_probing_dims_from_calibration,
)
from function.agreement_tasks import build_prompt_and_gold, evaluate_prediction, load_task_items
from function.perturb import (
    FrequencyHook,
    _apply_frequency_operation,
    _get_decoder_layers,
    _run_single_text,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_config_fingerprint(cfg: dict) -> str:
    material = {
        "model_name": cfg.get("model_name"),
        "sample_limit_per_task": cfg.get("sample_limit_per_task"),
        "layer_indices": cfg.get("layer_indices", []),
        "task_layer_overrides": cfg.get("task_layer_overrides", {}),
        "methods": cfg.get("methods", {}),
        "energy_alignment": cfg.get("energy_alignment", {}),
        "generation": cfg.get("generation", {}),
        "ppl": cfg.get("ppl", {}),
        "tasks": cfg.get("tasks", {}),
    }
    raw = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def resolve_task_layers(cfg: dict, task_name: str) -> List[int]:
    default_layers = [int(x) for x in cfg.get("layer_indices", [])]
    overrides = cfg.get("task_layer_overrides", {})
    task_layers = overrides.get(task_name, default_layers)
    dedup = []
    seen = set()
    for li in task_layers:
        li = int(li)
        if li in seen:
            continue
        seen.add(li)
        dedup.append(li)
    return dedup


def _layer_key(task_name: str, layer_index: int) -> Tuple[str, int]:
    return str(task_name), int(layer_index)


def _read_layer_record(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "task" not in data or "layer_index" not in data:
            return None
        return data
    except Exception:
        return None


def load_existing_layer_records(out_dir: str) -> Dict[Tuple[str, int], dict]:
    records: Dict[Tuple[str, int], dict] = {}
    if not os.path.isdir(out_dir):
        return records
    for name in os.listdir(out_dir):
        if not name.endswith(".json"):
            continue
        if "_layer_" not in name:
            continue
        rec = _read_layer_record(os.path.join(out_dir, name))
        if rec is None:
            continue
        records[_layer_key(rec["task"], int(rec["layer_index"]))] = rec
    return records


def _save_summary(rows_by_key: Dict[Tuple[str, int], dict], summary_path: str):
    rows = sorted(rows_by_key.values(), key=lambda x: (str(x["task"]), int(x["layer_index"])))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _contiguous_prefix_samples(samples: List[dict], target_limit: int) -> List[dict]:
    if not isinstance(samples, list):
        return []
    by_idx = {}
    for s in samples:
        try:
            idx = int(s.get("index", -1))
        except Exception:
            continue
        if idx < 0:
            continue
        by_idx[idx] = s
    out = []
    for i in range(max(0, target_limit)):
        if i not in by_idx:
            break
        out.append(by_idx[i])
    return out


def _infer_resume_state(
    existing_record: Optional[dict],
    target_limit: int,
    method_names: List[str],
) -> Tuple[int, Dict[str, List[dict]]]:
    kept: Dict[str, List[dict]] = {}
    if not isinstance(existing_record, dict):
        kept["baseline"] = []
        for m in method_names:
            kept[m] = []
        return 0, kept

    base_pref = _contiguous_prefix_samples(existing_record.get("baseline", {}).get("samples", []), target_limit)
    kept["baseline"] = base_pref
    counts = [len(base_pref)]

    for m in method_names:
        pref = _contiguous_prefix_samples(existing_record.get(m, {}).get("samples", []), target_limit)
        kept[m] = pref
        counts.append(len(pref))

    resume_n = min(counts) if counts else 0
    for k in list(kept.keys()):
        kept[k] = kept[k][:resume_n]
    return resume_n, kept


def maybe_set_eager_attention(model):
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    except Exception:
        pass


def _max_new_tokens_for_task(task_name: str, generation_cfg: dict) -> int:
    if task_name == "mawps":
        return int(generation_cfg.get("max_new_tokens_mawps", 64))
    if task_name == "boolq":
        return int(generation_cfg.get("max_new_tokens_boolq", generation_cfg.get("max_new_tokens_choice", 8)))
    if task_name == "ethics_commonsense":
        return int(generation_cfg.get("max_new_tokens_ethics", generation_cfg.get("max_new_tokens_choice", 8)))
    return int(generation_cfg.get("max_new_tokens_choice", 8))


def run_task_eval(
    model,
    tokenizer,
    device: str,
    task_name: str,
    items: List[dict],
    layer_index: int,
    method_name: str,
    hook: Optional[FrequencyHook],
    generation_cfg: dict,
    start_index: int = 0,
    existing_samples: Optional[List[dict]] = None,
) -> Dict[str, object]:
    prompts = []
    golds = []
    for item in items:
        p, g = build_prompt_and_gold(task_name, item)
        prompts.append(p)
        golds.append(g)
    target_len = activation._compute_uniform_token_length(tokenizer, prompts, fast_mode=True)
    target_len = min(target_len, 256)

    sample_outputs = list(existing_samples or [])
    correct = sum(1 for s in sample_outputs if bool(s.get("correct", False)))
    start = max(0, min(int(start_index), len(items)))
    for i in tqdm(range(start, len(items)), desc=f"{task_name} L{layer_index} {method_name}", leave=False):
        prompt = prompts[i]
        gold = golds[i]
        max_new = _max_new_tokens_for_task(task_name, generation_cfg)
        out = _run_single_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=prompt,
            max_length=target_len,
            layer_index=layer_index,
            hook=hook,
            topk=1,
            max_new_tokens=max_new,
            temperature=generation_cfg.get("temperature", 0.0),
            max_time=generation_cfg.get("max_time", None),
        )
        gen = (out.get("generated") or "").strip()
        pred = evaluate_prediction(task_name, gen, gold)
        correct += int(pred["correct"])
        sample_outputs.append(
            {
                "index": i,
                "prompt": prompt,
                "generated": gen,
                "pred": pred["pred"],
                "gold": pred["gold"],
                "correct": pred["correct"],
            }
        )
    acc = correct / max(1, len(items))
    return {"accuracy": acc, "samples": sample_outputs}


def collect_ppl_texts(dataset_path: str, split: str, limit: int) -> List[str]:
    ds = load_from_disk(dataset_path)[split]
    texts = []
    for item in ds:
        t = (item.get("text") or "").strip()
        if t:
            texts.append(t)
        if len(texts) >= limit:
            break
    return texts


def compute_ppl(
    model,
    tokenizer,
    device: str,
    texts: List[str],
    layer_index: int,
    hook: Optional[FrequencyHook],
    max_length: int,
) -> float:
    from function.perturb import register_layer_hook

    total_nll = 0.0
    total_tokens = 0.0
    ctx = register_layer_hook(model, layer_index, hook) if hook else nullcontext(layer_index)
    with torch.no_grad():
        with ctx:
            for text in tqdm(texts, desc=f"PPL L{layer_index}", leave=False):
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = model(**encoded, use_cache=False, return_dict=True)
                logits = outputs.logits[:, :-1, :]
                labels = encoded["input_ids"][:, 1:]
                mask = encoded["attention_mask"][:, 1:].float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(mask)
                total_nll += float((loss * mask).sum().item())
                total_tokens += float(mask.sum().item())
    if total_tokens <= 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


def _resolve_layer_index(model, layer_index: int) -> int:
    layers = _get_decoder_layers(model)
    if layer_index < 0:
        layer_index = len(layers) + layer_index
    return max(0, min(int(layer_index), len(layers) - 1))


def _resolve_method_ratio(config: dict, method_name: str, default: float = 0.3) -> float:
    mcfg = config["methods"][method_name]
    for key in ("r", "top_ratio", "random_dims_ratio", "dim_ratio"):
        if key in mcfg:
            return max(0.0, min(1.0, float(mcfg[key])))
    # backward-compatible auto mapping from dft band width
    mode = str(mcfg.get("topk_dims_mode", ""))
    if mode == "auto_from_band_width":
        dft_cfg = config["methods"].get("dft_noise", {})
        if "r" in dft_cfg:
            return max(0.0, min(1.0, float(dft_cfg["r"])))
        low = float(dft_cfg.get("band_low", 0.45))
        high = float(dft_cfg.get("band_high", 0.5))
        return max(0.0, min(1.0, (high - low) / 0.5))
    return max(0.0, min(1.0, float(default)))


def _resolve_method_topk(config: dict, method_name: str, hidden_size: int) -> int:
    mcfg = config["methods"][method_name]
    mode = str(mcfg.get("topk_dims_mode", "from_r"))
    if mode in {"from_r", "auto_from_band_width"}:
        ratio = _resolve_method_ratio(config, method_name, default=0.3)
        k = int(round(hidden_size * ratio))
        return max(1, min(hidden_size, k))
    return max(1, min(hidden_size, int(mcfg.get("topk_dims", 128))))


def _stable_seed(base_seed: int, *parts) -> int:
    text = "|".join(str(x) for x in parts)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(h, 16)


def _sample_random_dims(hidden_size: int, ratio: float, seed: int) -> List[int]:
    ratio = max(0.0, min(1.0, float(ratio)))
    k = max(1, min(hidden_size, int(round(hidden_size * ratio))))
    idx = list(range(hidden_size))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    return sorted(int(x) for x in idx[:k])


def _method_edit_kwargs(method_name: str, method_cfg: dict, dims_by_method: Dict[str, List[int]], noise_std: float):
    op_type = str(method_cfg.get("op_type", "act_noise"))
    if op_type == "band_noise":
        top_ratio = method_cfg.get("r", method_cfg.get("top_ratio", None))
        if top_ratio is not None:
            r = max(0.0, min(1.0, float(top_ratio)))
            low = 0.5 * (1.0 - r)
            high = 0.5
        else:
            low = float(method_cfg.get("band_low", 0.45))
            high = float(method_cfg.get("band_high", 0.5))
        return {
            "op_type": "band_noise",
            "dims": None,
            "noise_std": float(noise_std),
            "low_cutoff": float(low),
            "high_cutoff": float(high),
        }
    return {
        "op_type": op_type,
        "dims": dims_by_method.get(method_name, []),
        "noise_std": float(noise_std),
        "low_cutoff": None,
        "high_cutoff": None,
    }


def _collect_calibration_hidden_states(
    model,
    tokenizer,
    device: str,
    task_name: str,
    items: List[dict],
    layer_index: int,
    limit: int,
) -> List[torch.Tensor]:
    prompts = [build_prompt_and_gold(task_name, it)[0] for it in items[: max(1, int(limit))]]
    target_len = activation._compute_uniform_token_length(tokenizer, prompts, fast_mode=True)
    target_len = min(target_len, 256)
    resolved = _resolve_layer_index(model, int(layer_index))

    states: List[torch.Tensor] = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Energy calib {task_name} L{layer_index}", leave=False):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=target_len,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states[resolved + 1]
            valid_len = int(encoded["attention_mask"][0].sum().item())
            valid_len = max(valid_len, 1)
            states.append(hs[:, :valid_len, :].detach())
    return states


def _estimate_edit_energy_ratio(
    hidden_states: List[torch.Tensor],
    edit_kwargs: dict,
    seed: int,
) -> float:
    if not hidden_states:
        return 0.0
    torch.manual_seed(int(seed))
    ratios = []
    with torch.no_grad():
        for hs in hidden_states:
            edited = _apply_frequency_operation(
                hidden_states=hs,
                op_type=edit_kwargs["op_type"],
                dims=edit_kwargs.get("dims"),
                noise_std=float(edit_kwargs.get("noise_std", 1.0)),
                low_cutoff=edit_kwargs.get("low_cutoff"),
                high_cutoff=edit_kwargs.get("high_cutoff"),
                force_roundtrip=True,
            )
            delta = (edited.float() - hs.float()).pow(2).mean()
            base = hs.float().pow(2).mean().clamp_min(1e-12)
            ratios.append(float((delta / base).item()))
    return float(sum(ratios) / max(1, len(ratios)))


def compute_aligned_noise_stds(
    config: dict,
    task_name: str,
    layer_index: int,
    hidden_states: List[torch.Tensor],
    dims_by_method: Dict[str, List[int]],
) -> Dict[str, float]:
    methods_cfg = config["methods"]
    enabled = [name for name, m in methods_cfg.items() if m.get("enabled", True)]
    out = {name: float(methods_cfg[name].get("noise_std", 1.0)) for name in enabled}

    align_cfg = config.get("energy_alignment", {})
    if not bool(align_cfg.get("enabled", True)):
        return out
    if not hidden_states:
        return out

    reference_method = str(align_cfg.get("reference_method", "dft_noise"))
    if reference_method not in enabled:
        reference_method = enabled[0]

    ref_seed = int(config.get("seed", 42)) + 101 + int(layer_index)
    ref_kwargs = _method_edit_kwargs(
        reference_method,
        methods_cfg[reference_method],
        dims_by_method,
        out[reference_method],
    )
    ref_energy = _estimate_edit_energy_ratio(hidden_states, ref_kwargs, ref_seed)
    if ref_energy <= 0:
        return out

    min_std = float(align_cfg.get("min_noise_std", 1e-4))
    max_std = float(align_cfg.get("max_noise_std", 10.0))
    for idx, method_name in enumerate(enabled):
        if method_name == reference_method:
            continue
        base_std = float(methods_cfg[method_name].get("noise_std", 1.0))
        test_kwargs = _method_edit_kwargs(method_name, methods_cfg[method_name], dims_by_method, base_std)
        energy = _estimate_edit_energy_ratio(hidden_states, test_kwargs, ref_seed + 17 + idx)
        if energy <= 0:
            out[method_name] = base_std
            continue
        scaled = base_std * (ref_energy / energy) ** 0.5
        out[method_name] = float(max(min_std, min(max_std, scaled)))

    print(
        f"[EnergyAlign] {task_name} L{layer_index} ref={reference_method} "
        + ", ".join([f"{k}:{v:.4f}" for k, v in out.items()]),
        flush=True,
    )
    return out


def build_hooks_for_layer(config: dict, dims_by_method: Dict[str, List[int]], noise_stds: Dict[str, float]):
    hooks = {}
    for method_name, mcfg in config["methods"].items():
        if not mcfg.get("enabled", True):
            continue
        kwargs = _method_edit_kwargs(method_name, mcfg, dims_by_method, noise_stds.get(method_name, mcfg.get("noise_std", 1.0)))
        hooks[method_name] = FrequencyHook(
            op_type=kwargs["op_type"],
            dims=kwargs["dims"],
            noise_std=float(kwargs["noise_std"]),
            low_cutoff=kwargs["low_cutoff"],
            high_cutoff=kwargs["high_cutoff"],
            force_roundtrip=True,
        )
    return hooks


def plot_results(results: List[dict], out_dir: str):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_task: Dict[str, Dict[int, dict]] = {}
    for r in results:
        layer = int(r["layer_index"])
        task = str(r["task"])
        by_task.setdefault(task, {})[layer] = r

    method_names = []
    for r in results:
        for key, val in r.items():
            if key in {"task", "layer_index", "config_fingerprint", "baseline", "method_dims", "aligned_noise_std"}:
                continue
            if isinstance(val, dict) and "accuracy" in val:
                method_names.append(key)
    method_names = sorted(set(method_names))
    if "dft_noise" in method_names:
        method_names.remove("dft_noise")
        method_names = ["dft_noise"] + method_names

    for task, layer_rows in sorted(by_task.items(), key=lambda x: x[0]):
        layers = sorted(layer_rows.keys())
        if not layers:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for method in method_names:
            ys = []
            for li in layers:
                row = layer_rows.get(li, {})
                ys.append(float(row.get(method, {}).get("delta_acc", np.nan)))
            ax.plot(layers, ys, marker="o", linewidth=2.0, label=method)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Delta Accuracy (method - baseline)")
        ax.set_title(f"{task}: Delta ACC vs Layer")
        ax.set_xticks(layers)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{task}_delta_acc_vs_layer.png"), dpi=150)
        plt.close(fig)


def main():
    cfg_path = os.path.join("configs", "agreement_compare_workflow.json")
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "agreement_workflow_summary.json")

    resume_cfg = cfg.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", True))
    skip_if_exists = bool(resume_cfg.get("skip_if_layer_file_exists", True))
    require_fingerprint_match = bool(resume_cfg.get("require_fingerprint_match", True))
    save_summary_every_layer = bool(resume_cfg.get("save_summary_every_layer", True))
    run_fingerprint = compute_config_fingerprint(cfg)
    print(f"Run fingerprint: {run_fingerprint}", flush=True)
    target_sample_limit = int(cfg["sample_limit_per_task"])
    enabled_method_names = [name for name, mcfg in cfg["methods"].items() if mcfg.get("enabled", True)]
    reuse_existing_ppl = bool(resume_cfg.get("reuse_existing_ppl", True))

    print("Loading model...", flush=True)
    tokenizer, model, device = activation.load_model(cfg["model_name"])
    maybe_set_eager_attention(model)
    print("Model loaded.", flush=True)

    hidden_size = int(getattr(model.config, "hidden_size", 2048))
    topk_by_method = {}
    for method_name, mcfg in cfg["methods"].items():
        if not mcfg.get("enabled", True):
            continue
        selector = str(mcfg.get("selector", ""))
        if selector in {"diffmean", "probing"}:
            topk_by_method[method_name] = _resolve_method_topk(cfg, method_name, hidden_size)
    if topk_by_method:
        print("Resolved top-k dims: " + ", ".join([f"{k}={v}" for k, v in topk_by_method.items()]), flush=True)

    ppl_cfg = cfg.get("ppl", {})
    ppl_enabled = bool(ppl_cfg.get("enabled", False))
    ppl_texts = []
    if ppl_enabled:
        ppl_texts = collect_ppl_texts(
            ppl_cfg["dataset_path"],
            ppl_cfg.get("split", "test"),
            int(ppl_cfg.get("sample_limit", 50)),
        )
        print(f"PPL corpus loaded: {len(ppl_texts)} samples", flush=True)

    results_by_key: Dict[Tuple[str, int], dict] = {}
    if resume_enabled:
        raw_existing = load_existing_layer_records(out_dir)
        enabled_tasks = {t for t, tcfg in cfg["tasks"].items() if tcfg.get("enabled", True)}
        allowed_layers_by_task = {t: set(resolve_task_layers(cfg, t)) for t in enabled_tasks}
        for key, rec in raw_existing.items():
            task_name = str(rec.get("task", ""))
            layer_idx = int(rec.get("layer_index", -10**9))
            if task_name not in enabled_tasks:
                continue
            if layer_idx not in allowed_layers_by_task.get(task_name, set()):
                continue
            if require_fingerprint_match and str(rec.get("config_fingerprint", "")) != run_fingerprint:
                continue
            results_by_key[key] = rec
        print(
            f"Loaded existing layer records: {len(results_by_key)} (filtered from {len(raw_existing)})",
            flush=True,
        )

    tasks_cfg = cfg["tasks"]
    for task_name, tcfg in tqdm(tasks_cfg.items(), desc="Tasks"):
        if not tcfg.get("enabled", True):
            continue
        print(f"\n=== Task: {task_name} ===", flush=True)
        eval_items = load_task_items(
            tcfg["dataset_path"],
            tcfg["split"],
            int(cfg["sample_limit_per_task"]),
            int(cfg["seed"]),
        )

        task_layers = resolve_task_layers(cfg, task_name)
        print(f"Layer plan ({task_name}): {task_layers}", flush=True)
        for layer_index in tqdm(task_layers, desc=f"Layers({task_name})", leave=False):
            layer_key = _layer_key(task_name, int(layer_index))
            layer_out = os.path.join(out_dir, f"{task_name}_layer_{layer_index}.json")
            existing = results_by_key.get(layer_key)
            existing_fp_ok = isinstance(existing, dict) and str(existing.get("config_fingerprint", "")) == run_fingerprint
            resume_n = 0
            existing_samples_by_method: Dict[str, List[dict]] = {}
            can_resume_from_existing = existing is not None and (existing_fp_ok or not require_fingerprint_match)
            if resume_enabled and can_resume_from_existing:
                resume_n, existing_samples_by_method = _infer_resume_state(
                    existing_record=existing,
                    target_limit=target_sample_limit,
                    method_names=enabled_method_names,
                )
            if resume_enabled and skip_if_exists and os.path.exists(layer_out):
                fp_ok = existing is not None and str(existing.get("config_fingerprint", "")) == run_fingerprint
                fully_done = resume_n >= target_sample_limit
                if fully_done and ((not require_fingerprint_match) or fp_ok):
                    print(f"Skip existing layer: {task_name} L{layer_index}", flush=True)
                    continue
            if resume_n > 0:
                print(
                    f"Resume layer {task_name} L{layer_index}: reuse first {resume_n}/{target_sample_limit} samples",
                    flush=True,
                )

            dims_by_method: Dict[str, List[int]] = {}
            selector_methods = {}
            for _mname, _mcfg in cfg["methods"].items():
                if not _mcfg.get("enabled", True):
                    continue
                sel = str(_mcfg.get("selector", "")).lower()
                if sel in {"diffmean", "probing", "random"}:
                    selector_methods[_mname] = sel
            need_energy = bool(cfg.get("energy_alignment", {}).get("enabled", False))
            existing_method_dims = existing.get("method_dims", {}) if isinstance(existing, dict) else {}
            if existing_fp_ok and isinstance(existing_method_dims, dict):
                for method_name, vals in existing_method_dims.items():
                    if method_name not in selector_methods:
                        continue
                    if not isinstance(vals, list) or len(vals) == 0:
                        continue
                    dims_by_method[method_name] = [int(x) for x in vals]
                    print(f"Reuse existing method_dims for {task_name} L{layer_index} {method_name}", flush=True)

            calib_items = []
            calib_n = 0
            need_calibration = need_energy or any(sel in {"diffmean", "probing"} for sel in selector_methods.values())
            if need_calibration:
                for _mname, mcfg in cfg["methods"].items():
                    if not mcfg.get("enabled", True):
                        continue
                    calib_n = max(calib_n, int(mcfg.get("calibration_size", 0)))
                calib_n = max(calib_n, 8)
                calib_items = load_task_items(
                    tcfg["dataset_path"],
                    tcfg["split"],
                    max(calib_n, int(cfg["sample_limit_per_task"])),
                    int(cfg["seed"]) + 7,
                )

            missing_ranked = [m for m, sel in selector_methods.items() if sel in {"diffmean", "probing"} and m not in dims_by_method]
            if missing_ranked:
                x_calib, y_calib = collect_calibration_vectors_and_labels(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    task_name=task_name,
                    items=calib_items[:calib_n],
                    layer_index=int(layer_index),
                    generation_cfg=cfg["generation"],
                )
                for method_name in missing_ranked:
                    sel = selector_methods[method_name]
                    topk = int(topk_by_method.get(method_name, _resolve_method_topk(cfg, method_name, hidden_size)))
                    if sel == "diffmean":
                        dims_by_method[method_name] = select_diffmean_dims_from_calibration(
                            x=x_calib,
                            y=y_calib,
                            topk=topk,
                        )
                    elif sel == "probing":
                        mcfg = cfg["methods"][method_name]
                        dims_by_method[method_name] = select_probing_dims_from_calibration(
                            x=x_calib,
                            y=y_calib,
                            topk=topk,
                            steps=int(mcfg.get("probing_steps", 200)),
                            lr=float(mcfg.get("probing_lr", 0.1)),
                            weight_decay=float(mcfg.get("probing_weight_decay", 1e-4)),
                        )

            for method_name, mcfg in cfg["methods"].items():
                if not mcfg.get("enabled", True):
                    continue
                if method_name in dims_by_method:
                    continue
                sel = str(mcfg.get("selector", "")).lower()
                op_type = str(mcfg.get("op_type", "")).lower()
                if sel == "random" and op_type in {"act_zero", "act_noise"}:
                    ratio = _resolve_method_ratio(cfg, method_name, default=0.3)
                    dims_by_method[method_name] = _sample_random_dims(
                        hidden_size=hidden_size,
                        ratio=float(ratio),
                        seed=_stable_seed(int(cfg.get("seed", 42)), task_name, layer_index, method_name),
                    )

            noise_stds = {name: float(mcfg.get("noise_std", 1.0)) for name, mcfg in cfg["methods"].items() if mcfg.get("enabled", True)}
            if need_energy:
                energy_cfg = cfg.get("energy_alignment", {})
                energy_calib_size = int(energy_cfg.get("calibration_size", 8))
                energy_states = _collect_calibration_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    task_name=task_name,
                    items=calib_items,
                    layer_index=int(layer_index),
                    limit=energy_calib_size,
                )
                noise_stds = compute_aligned_noise_stds(
                    config=cfg,
                    task_name=task_name,
                    layer_index=int(layer_index),
                    hidden_states=energy_states,
                    dims_by_method=dims_by_method,
                )
            hooks = build_hooks_for_layer(cfg, dims_by_method, noise_stds)

            baseline_eval = run_task_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                task_name=task_name,
                items=eval_items,
                layer_index=int(layer_index),
                method_name="baseline",
                hook=None,
                generation_cfg=cfg["generation"],
                start_index=resume_n,
                existing_samples=existing_samples_by_method.get("baseline", []),
            )

            layer_record = {
                "task": task_name,
                "layer_index": int(layer_index),
                "config_fingerprint": run_fingerprint,
                "baseline": baseline_eval,
                "method_dims": {k: v for k, v in dims_by_method.items()},
                "aligned_noise_std": {k: float(v) for k, v in noise_stds.items()},
            }

            for method_name, hook in hooks.items():
                method_eval = run_task_eval(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    task_name=task_name,
                    items=eval_items,
                    layer_index=int(layer_index),
                    method_name=method_name,
                    hook=hook,
                    generation_cfg=cfg["generation"],
                    start_index=resume_n,
                    existing_samples=existing_samples_by_method.get(method_name, []),
                )
                method_result = {
                    "accuracy": method_eval["accuracy"],
                    "delta_acc": method_eval["accuracy"] - baseline_eval["accuracy"],
                    "samples": method_eval["samples"],
                }
                if ppl_enabled:
                    ppl = None
                    if reuse_existing_ppl and existing_fp_ok:
                        try:
                            ppl = float(existing.get(method_name, {}).get("ppl"))
                        except Exception:
                            ppl = None
                    if ppl is None:
                        ppl = compute_ppl(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            texts=ppl_texts,
                            layer_index=int(layer_index),
                            hook=hook,
                            max_length=int(ppl_cfg.get("max_length", 256)),
                        )
                    method_result["ppl"] = ppl
                layer_record[method_name] = method_result

            results_by_key[layer_key] = layer_record
            with open(layer_out, "w", encoding="utf-8") as f:
                json.dump(layer_record, f, ensure_ascii=False, indent=2)
            print(f"Saved {layer_out}", flush=True)

            if save_summary_every_layer:
                _save_summary(results_by_key, summary_path)
                print(f"Checkpoint summary saved: {summary_path}", flush=True)

    _save_summary(results_by_key, summary_path)
    all_results = sorted(results_by_key.values(), key=lambda x: (str(x["task"]), int(x["layer_index"])))
    plot_results(all_results, out_dir)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
