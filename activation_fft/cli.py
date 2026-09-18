"""Command line front end: `python -m activation_fft <command>`.

A thin layer over the library. Anything it can do is available as three or four
lines of Python, and the library is the better place to work from once you are
past looking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from . import (
    average_spectra,
    count_tokens,
    equal_power_split,
    hidden_states,
    load_model,
    plot_layer_sweep,
    plot_spectrum,
    token_spectrum,
)


def _as_text(item) -> str:
    """Pull a string out of one dataset entry.

    Accepts a bare string, or a dict under any of the keys these corpora tend to
    use. Lists of lines are joined, since a poem is usually stored that way.
    """
    if isinstance(item, dict):
        for key in ("text", "paragraphs", "content", "lines"):
            if key in item:
                item = item[key]
                break
        else:
            return ""
    if isinstance(item, list):
        return "\n".join(str(line) for line in item)
    return str(item)


def read_texts(args) -> List[str]:
    if args.text:
        return [args.text]

    path = Path(args.texts_file)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [item for group in data.values() for item in group]
        texts = [_as_text(item) for item in data]
    else:
        texts = path.read_text(encoding="utf-8").splitlines()

    texts = [t for t in texts if t.strip()]
    if not texts:
        raise SystemExit(f"no usable texts in {path}")
    return texts


def parse_layers(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def cmd_tokens(args) -> int:
    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    texts = read_texts(args)
    counts = count_tokens(lm, texts)
    print(f"{len(texts)} text(s), {min(counts)}-{max(counts)} tokens")
    for text, n in list(zip(texts, counts))[: args.limit]:
        preview = text.replace("\n", " / ")[:48]
        print(f"  {n:5d}  {preview}")
    return 0


def cmd_spectrum(args) -> int:
    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    texts = read_texts(args)
    layer = parse_layers(args.layers)[0]

    specs = []
    for result in hidden_states(lm, texts, layers=[layer], max_length=args.max_length):
        acts = result[layer]
        specs.append(token_spectrum(acts.values, layer=layer, window=args.window))

    lengths = {s.n_tokens for s in specs}
    if len(lengths) > 1:
        print(
            f"texts run {min(lengths)}-{max(lengths)} tokens, so their frequency axes differ; "
            "reporting the first only. Use equal-length texts to average.",
            file=sys.stderr,
        )
        spec = specs[0]
    else:
        spec = average_spectra(specs) if len(specs) > 1 else specs[0]

    period, freq, power = spec.peak(min_period=args.min_period, max_period=args.max_period)
    print(f"layer {layer}, {spec.n_tokens} tokens, {spec.n_samples} sample(s)")
    print(f"  bin spacing   {spec.resolution:.5f} cycles/token")
    print(f"  strongest     period {period:.2f} tokens/cycle (f={freq:.4f}), power {power:.4g}")
    for k, f, p in spec.harmonics(period):
        print(f"  harmonic {k}    f={f:.4f}  power {p:.4g}")

    if args.out:
        fig = plot_spectrum(spec, mark_period=args.mark_period, max_period=args.max_period or 32.0)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"  wrote {args.out}")
    return 0


def cmd_sweep(args) -> int:
    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    texts = read_texts(args)
    layers = parse_layers(args.layers) if args.layers else None

    per_layer = {}
    for result in hidden_states(lm, texts, layers=layers, max_length=args.max_length):
        for layer, acts in result.items():
            per_layer.setdefault(layer, []).append(
                token_spectrum(acts.values, layer=layer, window=args.window)
            )

    specs = []
    for layer in sorted(per_layer):
        group = per_layer[layer]
        if len({s.n_tokens for s in group}) > 1:
            specs.append(group[0])
        else:
            specs.append(average_spectra(group) if len(group) > 1 else group[0])

    print(f"{len(specs)} layers, {specs[0].n_tokens} tokens")
    for spec in specs:
        period, freq, power = spec.peak(min_period=args.min_period, max_period=args.max_period)
        print(f"  layer {spec.layer:3d}  peak period {period:6.2f}  power {power:.4g}")

    if args.out:
        fig = plot_layer_sweep(specs, mark_period=args.mark_period)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"  wrote {args.out}")
    return 0


def cmd_bands(args) -> int:
    lm = load_model(args.model, trust_remote_code=args.trust_remote_code)
    texts = read_texts(args)
    layer = parse_layers(args.layers)[0]

    acts = hidden_states(lm, texts[:1], layers=[layer], max_length=args.max_length)[0][layer]
    spec = token_spectrum(acts.values, layer=layer, window=args.window)
    try:
        pair = equal_power_split(spec, fraction=args.fraction)
    except ValueError as exc:
        print(f"no equal-power split available: {exc}", file=sys.stderr)
        return 1
    print(f"layer {layer}, {spec.n_tokens} tokens")
    print(f"  {pair}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="activation_fft",
        description="Token-axis Fourier analysis of language-model activations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p, needs_layer=True):
        p.add_argument("--model", required=True,
                       help="Hub id or local path, passed to transformers unchanged")
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--text", help="a single text to analyse")
        source.add_argument("--texts-file",
                            help="JSON list/dict of texts, or a file with one text per line")
        p.add_argument("--layers", default="-1" if needs_layer else None,
                       help="comma-separated layer indices; negative counts from the end")
        p.add_argument("--window", default="hann", choices=["hann", "hamming", "boxcar"])
        p.add_argument("--max-length", type=int, default=None, help="truncate texts to this many tokens")
        p.add_argument("--min-period", type=float, default=2.0)
        p.add_argument("--max-period", type=float, default=None)
        p.add_argument("--mark-period", type=float, default=None,
                       help="draw a reference line at this period")
        p.add_argument("--out", default=None, help="write a figure here")
        p.add_argument("--trust-remote-code", action="store_true")

    p = sub.add_parser("tokens", help="count tokens per text before assuming a period")
    p.add_argument("--model", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--texts-file")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_tokens)

    p = sub.add_parser("spectrum", help="spectrum of one layer")
    shared(p)
    p.set_defaults(func=cmd_spectrum)

    p = sub.add_parser("sweep", help="how the spectrum changes with depth")
    shared(p, needs_layer=False)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("bands", help="equal-power low/high band split for a fair ablation")
    shared(p)
    p.add_argument("--fraction", type=float, default=0.2,
                   help="share of total power each band should hold")
    p.set_defaults(func=cmd_bands)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
