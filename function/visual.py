import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_dft_plots(analysis_dir, image_dir, baseline_analysis_dir=None, use_period_axis=False):
    """
    Generate static Matplotlib plots for one dataset.
    - Plot mean power spectra for selected dimensions across categories and layers.
    - `baseline_analysis_dir` is ignored in this version and no difference plot is produced.
    - Images are saved under image/<dataset_name>/ with the dataset name and layer index.
    - When `use_period_axis` is True, the x-axis uses period (tokens/cycle) instead of frequency.
    """
    dataset_name = os.path.basename(os.path.normpath(analysis_dir))

    # Reset the image directory for this dataset.
    if os.path.exists(image_dir):
        try:
            shutil.rmtree(image_dir)
        except Exception:
            for root, dirs, files in os.walk(image_dir):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
    os.makedirs(image_dir, exist_ok=True)

    # Only traverse layer_* directories to avoid stale or unrelated folders.
    layer_folders = sorted([
        f for f in os.listdir(analysis_dir)
        if f.startswith('layer_') and os.path.isdir(os.path.join(analysis_dir, f))
    ])

    for layer_folder in tqdm(layer_folders, desc="Generating Matplotlib Plots"):
        layer_analysis_path = os.path.join(analysis_dir, layer_folder)
        analysis_files = glob.glob(os.path.join(layer_analysis_path, 'dft_analysis_*.json'))
        if not analysis_files:
            print(f"Skipping {layer_folder} because no analysis files were found.")
            continue

        for af in analysis_files:
            try:
                with open(af, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            category = data.get('category', '')
            m = re.match(rf'^(.*)_({re.escape(dataset_name)})$', category)
            base_category = m.group(1) if m else category

            freq = np.array(data.get('frequency_axis', []))
            mean_sel = np.array(data.get('spectra_selected', []))
            std_sel = np.array(data.get('std_selected', []))
            selected = data.get('selected_indices', [])

            if not (freq.size and mean_sel.size):
                continue

            # Trim the lowest frequency region to stay consistent with the analysis stage.
            freq = freq[7:]
            mean_sel = mean_sel[:, 7:]
            std_sel = std_sel[:, 7:]

            # Choose either frequency or period for the x-axis.
            if use_period_axis:
                # The low-frequency slice has already been removed, but keep a safety check.
                safe_freq = np.where(freq == 0, np.nan, freq)
                x_axis = 1.0 / safe_freq
                x_label = 'Period (tokens/cycle)'
            else:
                x_axis = freq
                x_label = 'Frequency (cycles/token)'

            # Extract the numeric layer index for the plot title and filename.
            m_layer = re.search(r"layer[ _](\d+)", layer_folder)
            layer_number = m_layer.group(1) if m_layer else layer_folder

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(mean_sel.shape[0]):
                mean = mean_sel[i]
                color = colors[i % len(colors)]
                ax.plot(x_axis, mean, color=color, label=f'Dim {selected[i]}')

            ax.set_title(f'{dataset_name} - Layer {layer_number}: {base_category}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Power Spectrum')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

            out_ds = os.path.join(image_dir, f'{dataset_name}_{layer_number}.png')
            fig.savefig(out_ds, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"All required PNG plots generated in {image_dir}")

if __name__ == '__main__':
    # Example direct run.
    generate_dft_plots(analysis_dir='output/analysis/cn_peoms', image_dir='output/image/cn_peoms')
