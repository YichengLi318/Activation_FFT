"""Which layers depend on which frequency band?

Delete a band of one layer's spectrum, let the rest of the model run on the
damaged signal, and score it. Repeat for every layer. The shape of the resulting
curve says where the model uses what the spectrum found.

    python run_locality.py --model <model> --dataset dataset/your_task.json

The low and high bands are chosen to hold equal power, so the comparison is
about what a band carries rather than how much energy deleting it removed. See
the repository README for why that matters.

Dataset format: a JSON list of {"prompt": ..., "answer": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))

from activation_fft import (  # noqa: E402
    SpectralEdit,
    average_spectra,
    equal_power_split,
    hidden_states,
    load_model,
    spectral_edit,
    token_spectrum,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Hub id or local path")
    p.add_argument("--dataset", required=True, help="JSON list of {prompt, answer}")
    p.add_argument("--layers", default="", help="comma-separated; default is every 4th layer")
    p.add_argument("--fraction", type=float, default=0.15,
                   help="share of total power each band should hold")
    p.add_argument("--limit", type=int, default=100, help="prompts to score per condition")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--out", default=str(HERE / "locality_curve.png"))
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_items(path, limit):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(str(d["prompt"]), str(d["answer"])) for d in items[:limit]]


def pad_length(tokenizer, prompts):
    """One padded length for every prompt, so the frequency axis is shared."""
    lengths = [len(tokenizer(p, padding=False, truncation=False)["input_ids"]) for p in prompts]
    target = min(max(lengths), 256)
    return target + (-target % 8)


def score(lm, items, target_len, layer, edit, max_new_tokens):
    """Fraction of prompts answered correctly, with `edit` attached to `layer`."""
    correct = 0
    context = spectral_edit(lm.model, layer, edit) if edit else None
    handle = context.__enter__() if context else None
    try:
        for prompt, answer in items:
            encoded = lm.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=target_len)
            # generate() rejects keys it does not recognise, and some tokenizers
            # hand back token_type_ids. A causal LM needs these two.
            batch = {k: encoded[k].to(lm.device) for k in ("input_ids", "attention_mask")}
            if edit is not None:
                # Keep the edit inside the prompt; pad positions are not signal.
                edit.n_tokens = int(batch["attention_mask"][0].sum())
            with torch.no_grad():
                out = lm.model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
                    pad_token_id=lm.tokenizer.pad_token_id,
                )
            new = out[0, batch["input_ids"].shape[1]:]
            text = lm.tokenizer.decode(new, skip_special_tokens=True).strip().lower()
            correct += text.startswith(answer.strip().lower())
    finally:
        if context:
            context.__exit__(None, None, None)
    return correct / max(len(items), 1)


def bands_for_layer(lm, prompts, layer, fraction):
    """Equal-power low and high cutoffs, measured on this layer's own spectrum."""
    specs = [
        token_spectrum(r[layer].values, layer=layer)
        for r in hidden_states(lm, prompts, layers=[layer])
    ]
    same = [s for s in specs if s.n_tokens == specs[0].n_tokens]
    spectrum = average_spectra(same) if len(same) > 1 else same[0]
    return equal_power_split(spectrum, fraction=fraction)


def main():
    args = parse_args()
    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    items = load_items(args.dataset, args.limit)
    prompts = [p for p, _ in items]
    target_len = pad_length(lm.tokenizer, prompts)

    layers = ([int(x) for x in args.layers.replace(" ", "").split(",") if x]
              or list(range(0, lm.n_layers, 4)))
    print(f"{lm.name}: {lm.n_layers} layers | {len(items)} prompts | padded to {target_len}")

    baseline = score(lm, items, target_len, layers[0], None, args.max_new_tokens)
    print(f"baseline accuracy {baseline:.3f}\n")

    rows = []
    for layer in layers:
        try:
            pair = bands_for_layer(lm, prompts[:8], layer, args.fraction)
        except ValueError as exc:
            print(f"layer {layer:3d}  no equal-power split: {exc}")
            continue

        # "none" applies the rFFT/irFFT round trip and nothing else, so the
        # ablations are measured against it rather than against the clean model.
        control = score(lm, items, target_len, layer, SpectralEdit(op="none"), args.max_new_tokens)
        low = score(lm, items, target_len, layer,
                    SpectralEdit(op="bandstop", low_cutoff=pair.low[0], high_cutoff=pair.low[1]),
                    args.max_new_tokens)
        high = score(lm, items, target_len, layer,
                     SpectralEdit(op="bandstop", low_cutoff=pair.high[0], high_cutoff=pair.high[1]),
                     args.max_new_tokens)

        rows.append({"layer": layer, "control": control, "drop_low": control - low,
                     "drop_high": control - high, "width_ratio": pair.width_ratio})
        print(f"layer {layer:3d}  control {control:.3f} | "
              f"drop low {control - low:+.3f} | drop high {control - high:+.3f} | "
              f"high band {pair.width_ratio:.1f}x wider")

    if not rows:
        raise SystemExit("no layer produced a usable band split; try a smaller --fraction")

    xs = [r["layer"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [r["drop_low"] for r in rows], "o-", color="#1f77b4", label="drop low band")
    ax.plot(xs, [r["drop_high"] for r in rows], "o-", color="#d62728", label="drop high band")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy drop vs round-trip control")
    ax.set_title(f"{Path(args.dataset).stem}: equal-power band ablation "
                 f"({args.fraction:.0%} of power each)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    summary = os.path.splitext(args.out)[0] + ".json"
    Path(summary).write_text(json.dumps({"baseline": baseline, "rows": rows}, indent=2),
                             encoding="utf-8")
    print(f"\nwrote {args.out} and {summary}")


if __name__ == "__main__":
    main()
