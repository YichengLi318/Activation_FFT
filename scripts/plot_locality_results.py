import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_ALIASES = {
    "race": ("local", "Local Task (RACE)"),
    "mctest": ("local", "Local Task (MCTest)"),
    "ag_news": ("global", "Global Task (AG News)"),
    "blimp": ("blimp", "BLiMP"),
    "number_count": ("number_count", "Number Count"),
}

METHOD_STYLE_MAP = {
    "highpass_drop_low20": ("Drop Low 20%", "#1f77b4"),
    "highpass_drop_low40": ("Drop Low 40%", "#4c78a8"),
    "lowpass_drop_high20": ("Drop High 20%", "#d62728"),
    "lowpass_drop_high40": ("Drop High 40%", "#f58518"),
    "drop_low_band_0p05_0p15": ("Drop Low Band 0.05-0.15", "#1f77b4"),
    "drop_high_band_0p35_0p50": ("Drop High Band 0.35-0.50", "#d62728"),
    "drop_high_band_0p40_0p50": ("Drop High Band 0.40-0.50", "#d62728"),
}
# High-contrast discrete palette for band-scan plots.
DEFAULT_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot locality experiment curves from summary JSON.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to locality summary JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_results(summary: dict):
    if isinstance(summary, dict) and "results" in summary and isinstance(summary["results"], dict):
        return summary["results"]
    return summary


def collect_task_data(summary: dict):
    task_layers = defaultdict(dict)
    records = unwrap_results(summary)
    for _, method_map in records.items():
        if not isinstance(method_map, dict):
            continue
        for method_name, record in method_map.items():
            if not isinstance(record, dict):
                continue
            task = str(record.get("task"))
            if task not in TASK_ALIASES:
                continue
            layer = int(record.get("layer"))
            baseline_acc = float(record.get("baseline_acc", 0.0))
            perturbed_acc = float(record.get("perturbed_acc", 0.0))
            acc_drop = baseline_acc - perturbed_acc
            task_layers[task].setdefault(layer, {})[method_name] = acc_drop
    return task_layers


def _parse_method_bounds(method_name: str):
    m = re.match(r"drop_band_(\d+)p(\d+)_(\d+)p(\d+)$", method_name)
    if m:
        lo = float(f"{m.group(1)}.{m.group(2)}")
        hi = float(f"{m.group(3)}.{m.group(4)}")
        return lo, hi
    m = re.match(r"(?:stop|noise)_band_(\d+)p(\d+)_(\d+)p(\d+)$", method_name)
    if m:
        lo = float(f"{m.group(1)}.{m.group(2)}")
        hi = float(f"{m.group(3)}.{m.group(4)}")
        return lo, hi
    m = re.match(r"drop_(low|high)_band_(\d+)p(\d+)_(\d+)p(\d+)$", method_name)
    if m:
        lo = float(f"{m.group(2)}.{m.group(3)}")
        hi = float(f"{m.group(4)}.{m.group(5)}")
        return lo, hi
    return None


def _method_sort_key(method_name: str):
    bounds = _parse_method_bounds(method_name)
    if method_name.startswith("stop_band_"):
        prefix_rank = 0
    elif method_name.startswith("noise_band_"):
        prefix_rank = 1
    elif method_name.startswith("drop_band_"):
        prefix_rank = 2
    else:
        prefix_rank = 3
    if bounds is not None:
        return (0, prefix_rank, bounds[0], bounds[1], method_name)
    return (1, prefix_rank, method_name)


def _method_label(method_name: str) -> str:
    if method_name in METHOD_STYLE_MAP:
        return METHOD_STYLE_MAP[method_name][0]
    bounds = _parse_method_bounds(method_name)
    if bounds is not None:
        if method_name.startswith("stop_band_"):
            return f"Stop Band {bounds[0]:.2f}-{bounds[1]:.2f}"
        if method_name.startswith("noise_band_"):
            return f"Noise Band {bounds[0]:.2f}-{bounds[1]:.2f}"
        return f"Drop Band {bounds[0]:.2f}-{bounds[1]:.2f}"
    return method_name


def _method_linestyle(method_name: str) -> str:
    if method_name.startswith("stop_band_"):
        return "-"
    if method_name.startswith("noise_band_"):
        return "--"
    return "-"


def _method_family(method_name: str) -> str:
    if method_name.startswith("stop_band_"):
        return "stop"
    if method_name.startswith("noise_band_"):
        return "noise"
    if method_name.startswith("drop_band_"):
        return "drop"
    return "other"


def resolve_method_specs(layer_map: dict):
    ordered = []
    seen = set()
    for layer in sorted(layer_map.keys()):
        for method_name in layer_map[layer].keys():
            if method_name == "baseline" or method_name in seen:
                continue
            seen.add(method_name)
            ordered.append(method_name)
    ordered = sorted(ordered, key=_method_sort_key)

    specs = []
    for idx, method_name in enumerate(ordered):
        if method_name in METHOD_STYLE_MAP:
            label, color = METHOD_STYLE_MAP[method_name]
        else:
            label = _method_label(method_name)
            color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        specs.append((method_name, label, color))
    return specs


def _plot_one_figure(short_name: str, title: str, layers: list, method_specs: list, layer_map: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method_name, label, color in method_specs:
        ys = [layer_map[layer].get(method_name, np.nan) for layer in layers]
        ax.plot(
            layers,
            ys,
            marker="o",
            linewidth=2.0,
            linestyle=_method_linestyle(method_name),
            color=color,
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_task(task_name: str, layer_map: dict, out_dir: Path):
    if task_name in TASK_ALIASES:
        short_name, title = TASK_ALIASES[task_name]
    else:
        short_name = task_name
        title = task_name.replace("_", " ").title()
    layers = sorted(layer_map.keys())
    method_specs = resolve_method_specs(layer_map)
    families = {}
    for spec in method_specs:
        families.setdefault(_method_family(spec[0]), []).append(spec)

    split_stop_noise = "stop" in families and "noise" in families
    if split_stop_noise:
        combined_path = out_dir / f"{short_name}_acc_drop_vs_layer.png"
        if combined_path.exists():
            combined_path.unlink()
        for family in ["stop", "noise"]:
            fam_specs = families.get(family, [])
            if not fam_specs:
                continue
            family_title = f"{title} ({family.title()})"
            out_path = out_dir / f"{short_name}_{family}_acc_drop_vs_layer.png"
            _plot_one_figure(short_name, family_title, layers, fam_specs, layer_map, out_path)
        return

    out_path = out_dir / f"{short_name}_acc_drop_vs_layer.png"
    _plot_one_figure(short_name, title, layers, method_specs, layer_map, out_path)


def main():
    args = parse_args()
    summary = load_summary(Path(args.summary))
    task_data = collect_task_data(summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_name, layer_map in task_data.items():
        plot_task(task_name, layer_map, out_dir)

    print(f"Plotted {len(task_data)} tasks into {out_dir}")


if __name__ == "__main__":
    main()
