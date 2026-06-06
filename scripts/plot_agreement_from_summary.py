import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot delta-accuracy curves from agreement summary JSON.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to agreement summary JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for _, methods in data.items():
            if not isinstance(methods, dict):
                continue
            rows.append(methods)
    else:
        raise ValueError("Unsupported summary format.")
    return rows


def collect_task_layer_table(rows):
    by_task = defaultdict(dict)
    method_names = set()
    for methods in rows:
        if not isinstance(methods, dict):
            continue
        for method_name, rec in methods.items():
            if not isinstance(rec, dict) or "task" not in rec or "layer" not in rec:
                continue
            task = str(rec["task"])
            layer = int(rec["layer"])
            by_task[task].setdefault(layer, {})
            by_task[task][layer][method_name] = float(rec.get("delta_acc", np.nan))
            method_names.add(method_name)
    method_names = sorted(method_names)
    if "dft_noise" in method_names:
        method_names.remove("dft_noise")
        method_names = ["dft_noise"] + method_names
    return by_task, method_names


def plot_task_curves(by_task, method_names, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, layer_map in sorted(by_task.items(), key=lambda x: x[0]):
        layers = sorted(layer_map.keys())
        fig, ax = plt.subplots(figsize=(8, 4))
        for method in method_names:
            ys = [layer_map[li].get(method, np.nan) for li in layers]
            ax.plot(layers, ys, marker="o", linewidth=2.0, label=method)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Delta Accuracy (method - baseline)")
        ax.set_title(f"{task}: Delta ACC vs Layer")
        ax.set_xticks(layers)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{task}_delta_acc_vs_layer.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(Path(args.summary))
    by_task, method_names = collect_task_layer_table(rows)
    plot_task_curves(by_task, method_names, Path(args.out_dir))
    print(f"Plotted {len(by_task)} tasks, methods={method_names}")


if __name__ == "__main__":
    main()
