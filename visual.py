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
    使用 Matplotlib 生成静态图（按数据集分目录）：
    - 对单个数据集的各类别、各层绘制选定维度的功率谱均值（不绘制标准差阴影）。
    - baseline_analysis_dir 参数将被忽略（不绘制差分）。
    - 图像按 image/<dataset_name>/ 保存，文件名包含数据集名与层号。
    - 新增 use_period_axis：为 True 时，横轴用周期（tokens/cycle），否则为频率（cycles/token）。
    """
    dataset_name = os.path.basename(os.path.normpath(analysis_dir))

    # 清空该数据集的图像目录
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

    # 仅遍历 layer_* 目录，避免误读历史目录
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

            # 频率裁剪（去掉最低频段，与分析阶段保持一致）
            freq = freq[7:]
            mean_sel = mean_sel[:, 7:]
            std_sel = std_sel[:, 7:]

            # 选择横轴：频率或周期
            if use_period_axis:
                # 由于已裁剪低频，freq 不含 0；为稳健起见仍做保护
                safe_freq = np.where(freq == 0, np.nan, freq)
                x_axis = 1.0 / safe_freq
                x_label = 'Period (tokens/cycle)'
            else:
                x_axis = freq
                x_label = 'Frequency (cycles/token)'

            # 提取层号
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
    # Example direct run (kept in sync with new output paths)
    generate_dft_plots(analysis_dir='output/analysis/cn_peoms', image_dir='output/image/cn_peoms')