import os
import json
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_token_label(tok: str) -> str:
    tok = tok.replace("\n", "\\n")
    if len(tok) > 18:
        tok = tok[:15] + "..."
    return tok


def _plot_single_sample(sample_path, out_path):
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline = data.get("baseline", {})
    perturbed = data.get("perturbed", {})
    base_topk = baseline.get("topk", [])
    pert_topk = perturbed.get("topk", [])

    # Merge token labels while preserving baseline order first.
    token_order = []
    seen = set()
    for item in base_topk + pert_topk:
        t = item.get("token", "")
        if t not in seen:
            seen.add(t)
            token_order.append(t)

    base_probs = []
    pert_probs = []
    for tok in token_order:
        b = next((x["prob"] for x in base_topk if x.get("token") == tok), 0.0)
        p = next((x["prob"] for x in pert_topk if x.get("token") == tok), 0.0)
        base_probs.append(b)
        pert_probs.append(p)

    x = range(len(token_order))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], base_probs, width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], pert_probs, width=width, label="perturbed")

    labels = [_safe_token_label(tok) for tok in token_order]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"{data.get('dataset','')} | {data.get('category','')} | layer {data.get('layer_index','?')} | op {data.get('operation','')}"
    )
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_perturb_topk_plots(perturb_root, image_root, samples_per_category=3):
    """
    Plot baseline vs perturbed top-k probabilities from frequency-edit outputs.
    - perturb_root: a directory such as output/perturbation/<dataset>
    - image_root: output image root, for example output/image_perturb
    - samples_per_category: maximum number of samples to plot per category
    """
    if not os.path.isdir(perturb_root):
        raise FileNotFoundError(f"Perturbation root not found: {perturb_root}")

    # Accept either a dataset root or a specific operation-layer directory.
    subdirs = [d for d in os.listdir(perturb_root) if os.path.isdir(os.path.join(perturb_root, d))]
    op_dirs = []
    if subdirs and not glob.glob(os.path.join(perturb_root, "sample_*.json")):
        # Treat the input as a dataset root.
        op_dirs = [os.path.join(perturb_root, d) for d in subdirs]
        dataset_name = os.path.basename(os.path.normpath(perturb_root))
    else:
        # Treat the input as an operation-layer directory.
        op_dirs = [perturb_root]
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(perturb_root)))

    for op_dir in op_dirs:
        op_tag = os.path.basename(op_dir)
        categories = [d for d in os.listdir(op_dir) if os.path.isdir(os.path.join(op_dir, d))]
        for cat in categories:
            cat_dir = os.path.join(op_dir, cat)
            samples = sorted(glob.glob(os.path.join(cat_dir, "sample_*.json")))
            samples = samples[:samples_per_category]
            for sp in samples:
                fn = os.path.splitext(os.path.basename(sp))[0]
                out_path = os.path.join(image_root, dataset_name, op_tag, cat, f"{fn}.png")
                _plot_single_sample(sp, out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Example: generate baseline-vs-perturbed visualizations for tiny_story_test.
    generate_perturb_topk_plots(
        perturb_root="output/perturbation/tiny_story_test",
        image_root="output/image_perturb",
        samples_per_category=3,
    )
