"""Does a layer's spectrum know how long a poetic line is?

Regulated Chinese verse puts exactly five or seven characters in every line and
repeats it for the whole poem, so the token sequence carries a period you know
before you look. If the strongest period matches, the transform is measuring
what it claims to.

    python run_poem_spectrum.py --model <model> --poems dataset/poems.json

Dataset format, one group per line length:

    {"five_char":  [{"lines": ["白日依山尽", "黄河入海流", ...]}, ...],
     "seven_char": [{"lines": ["朝辞白帝彩云间", ...]}, ...]}

Supply your own corpus; dataset/ ships empty. The script prints measured tokens
per line before any peak, because that number is what the experiment rests on:
a character is not guaranteed to be one token, and a line separator is a token
too. Try --separator '，' to see the period move.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))

from activation_fft import (  # noqa: E402
    average_spectra,
    count_tokens,
    hidden_states,
    load_model,
    min_tokens_to_separate,
    plot_layer_sweep,
    token_spectrum,
)

EXPECTED_PERIOD = {"five_char": 5, "seven_char": 7}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="Hub id or local path")
    p.add_argument("--poems", default=str(HERE / "dataset" / "poems.json"))
    p.add_argument("--layers", default="0,4,8,12,16,20")
    p.add_argument("--separator", default="", help="text between lines; try '，'")
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def tokens_per_line(lm, category, poems, separator):
    """Measure the period to look for, which may not be the one in the name."""
    lines = [line + separator for poem in poems for line in poem["lines"]]
    counts = Counter(count_tokens(lm, lines))
    modal, n = counts.most_common(1)[0]
    total = sum(counts.values())
    print(f"  tokens per line {dict(sorted(counts.items()))}, modal {modal} "
          f"({n}/{total} lines)")

    named = EXPECTED_PERIOD.get(category)
    if named and modal != named:
        print(f"    named for {named} characters but tokenizes to {modal} positions, "
              f"so look for period {modal}")
    if n / total < 0.9:
        print("    line length too inconsistent for one period; check the separator")
    return modal


def spectra_by_layer(lm, poems, layers, separator):
    """Spectra per layer, averaged over the largest group of equal-length poems."""
    texts = [separator.join(poem["lines"]) for poem in poems]
    groups = {}
    for text, n in zip(texts, count_tokens(lm, texts)):
        groups.setdefault(n, []).append(text)
    n_tokens, group = max(groups.items(), key=lambda kv: len(kv[1]))
    print(f"    {len(group)}/{len(texts)} poems at {n_tokens} tokens "
          f"(lengths {sorted(groups)})")

    floor = min_tokens_to_separate(5, 7)
    if n_tokens < floor:
        print(f"    {n_tokens} tokens cannot separate period 5 from 7; {floor} is the minimum")

    per_layer = {layer: [] for layer in layers}
    for result in hidden_states(lm, group, layers=layers):
        for layer, acts in result.items():
            per_layer[layer].append(token_spectrum(acts.values, layer=layer))
    return {
        layer: average_spectra(s) if len(s) > 1 else s[0] for layer, s in per_layer.items()
    }, n_tokens


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.replace(" ", "").split(",") if x]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.poems).exists():
        raise SystemExit(f"{args.poems} not found. dataset/ ships empty; see the docstring "
                         "for the format.")

    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    print(f"{lm.name}: {lm.n_layers} layers, hidden size {lm.hidden_size}")
    print(f"lines joined by {args.separator!r}\n")

    corpus = json.loads(Path(args.poems).read_text(encoding="utf-8"))
    for category, poems in corpus.items():
        if category.startswith("_") or not isinstance(poems, list):
            continue
        print(f"[{category}] {len(poems)} poems")
        period = tokens_per_line(lm, category, poems, args.separator)
        by_layer, n_tokens = spectra_by_layer(lm, poems, layers, args.separator)
        print(f"    bin spacing {1 / n_tokens:.4f} cycles/token; "
              f"period {period} sits at f={1 / period:.4f}")

        for layer in sorted(by_layer):
            spec = by_layer[layer]
            peak, _, power = spec.peak(min_period=2.0, max_period=n_tokens / 3)
            verdict = "matches" if abs(peak - period) <= 1.0 else "no"
            print(f"    layer {layer:3d}  peak period {peak:6.2f}  power {power:9.4g}  "
                  f"{verdict} ({period})")

        fig = plot_layer_sweep(
            [by_layer[layer] for layer in sorted(by_layer)],
            mark_period=period,
            title=f"{category}: layer sweep, {n_tokens} tokens",
        )
        path = out_dir / f"{category}_layer_sweep.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    wrote {path.name}\n")


if __name__ == "__main__":
    main()
