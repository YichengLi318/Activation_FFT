import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from tqdm import tqdm
import glob

# 1. 载入模型
model_path = os.path.join(os.path.dirname(__file__), 'model', 'gpt2-large')

print(f"Loading model from '{model_path}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # 将模型设置为评估模式
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    print(f"Please ensure the model files are downloaded and placed in '{model_path}'.")
    exit()

# 探测模型属性
num_layers = model.config.n_layer
hidden_size = model.config.n_embd
print(f"Model '{model_path}' has {num_layers} hidden layers.")
print(f"Activation vector length (hidden size): {hidden_size}")
layers_to_analyze = range(num_layers)
print(f"Will analyze all {num_layers} layers.")

# 统一的输出文件夹
base_output_dir = 'output'
output_dir_images_3d = os.path.join(base_output_dir, 'images', '3d_spectrum_analysis')
output_dir_data = os.path.join(base_output_dir, 'data')

os.makedirs(output_dir_images_3d, exist_ok=True)
os.makedirs(output_dir_data, exist_ok=True)

print(f"Output will be saved to the '{base_output_dir}' directory.")

# 2. 测试样例
syntax_pairs = [
    {
        "name": "Subject-Verb-Agreement",
        "A": "The key to the doors is lost.",
        "B": "The key to the doors are lost.",
        "target_word_A": "is",
        "target_word_B": "are"
    },
    {
        "name": "Determiner-Noun-Agreement",
        "A": "She ate an apple.",
        "B": "She ate a apple.",
        "target_word_A": "an",
        "target_word_B": "a"
    },
    {
        "name": "Third-Person-Singular",
        "A": "He writes a letter every day.",
        "B": "He write a letter every day.",
        "target_word_A": "writes",
        "target_word_B": "write"
    },
    {
        "name": "Past-Tense-Regular",
        "A": "Yesterday I walked to the park.",
        "B": "Yesterday I walk to the park.",
        "target_word_A": "walked",
        "target_word_B": "walk"
    }
]

# 3. 激活提取
def get_activations(text, layer_idx):
    """为给定的文本和层索引提取激活值。"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states 元组的第一个元素是词嵌入输出，所以层数需要+1
    activations = outputs.hidden_states[layer_idx + 1]
    return inputs, activations.squeeze(0)  # 移除批次维度

def find_target_token_index(inputs, word):
    """在 token 列表中找到目标词的索引。"""
    token_ids = inputs['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    possible_tokens = [tokenizer.tokenize(' ' + word)[0], tokenizer.tokenize(word)[0]]
    for i, token in enumerate(tokens):
        if token in possible_tokens:
            return i
            
    print(f"Warning: Could not find token for target word '{word}' in '{tokenizer.decode(token_ids)}'.")
    print(f"Tokenized sentence: {tokens}")
    return None

def analyze_frequency_bands(freq, dft_data):
    """Calculate the average magnitude in predefined frequency bands."""
    bands = {
        "low": (0, 0.1),
        "mid-low": (0.1, 0.2),
        "mid": (0.2, 0.3),
        "mid-high": (0.3, 0.4),
        "high": (0.4, 0.5)
    }
    band_magnitudes = {}
    for name, (low_freq, high_freq) in bands.items():
        indices = np.where((freq >= low_freq) & (freq < high_freq))
        if len(indices[0]) > 0:
            band_magnitudes[name] = np.mean(dft_data[indices])
        else:
            band_magnitudes[name] = 0
    return band_magnitudes

# 创建一个包含所有任务的列表
tasks = [(pair, layer) for pair in syntax_pairs for layer in layers_to_analyze]

print("\n--- Starting Analysis ---")
for pair, layer_to_extract in tqdm(tasks, desc="Analyzing Layers"):
    pair_name = pair["name"]
    
    json_filename = os.path.join(output_dir_data, f"{pair_name}_layer_{layer_to_extract}.json")
    if os.path.exists(json_filename):
        continue # 如果文件已存在，则跳过

    sentence_A, sentence_B = pair["A"], pair["B"]
    target_A, target_B = pair["target_word_A"], pair["target_word_B"]

    inputs_A, activations_A = get_activations(sentence_A, layer_to_extract)
    inputs_B, activations_B = get_activations(sentence_B, layer_to_extract)

    if activations_A.shape[0] != activations_B.shape[0]:
        print(f"Warning: Token count mismatch for '{pair_name}' at layer {layer_to_extract}. Skipping.")
        continue

    target_idx_A = find_target_token_index(inputs_A, target_A)
    target_idx_B = find_target_token_index(inputs_B, target_B)

    if target_idx_A is None or target_idx_B is None:
        print(f"Warning: Could not find target word for '{pair_name}' at layer {layer_to_extract}. Skipping.")
        continue

    activation_vec_A = activations_A[target_idx_A].numpy()
    activation_vec_B = activations_B[target_idx_B].numpy()

    # 4. 对激活向量进行DFT
    dft_A = np.abs(np.fft.fft(activation_vec_A))
    dft_B = np.abs(np.fft.fft(activation_vec_B))
    dft_diff = dft_A - dft_B

    N = dft_A.shape[-1]
    freq = np.fft.fftfreq(N, d=1.0)[:N // 2]
    dft_diff_pos = dft_diff[:N // 2]

    dft_all_A = np.abs(np.fft.fft(activations_A.numpy(), axis=1))
    dft_all_B = np.abs(np.fft.fft(activations_B.numpy(), axis=1))
    avg_dft_diff_pos = np.mean(dft_all_A - dft_all_B, axis=0)[:N // 2]

    target_bands = analyze_frequency_bands(freq, dft_diff_pos)
    avg_bands = analyze_frequency_bands(freq, avg_dft_diff_pos)

    # 准备要保存的数据
    layer_results = {
        "pair_name": pair_name,
        "layer": layer_to_extract,
        "dft_difference": dft_diff_pos.tolist(),
        "avg_dft_difference_all_tokens": avg_dft_diff_pos.tolist(),
        "band_analysis_target_word": target_bands,
        "band_analysis_average": avg_bands
    }

    # 保存独立的JSON文件
    with open(json_filename, 'w') as f:
        json.dump(layer_results, f, indent=4)

# 5. 将结果可视化并保存

print("\n--- Analysis complete. Loading data for 3D plotting... ---")

# 从保存的JSON文件中加载所有数据
all_results_data = {}
json_files = glob.glob(os.path.join(output_dir_data, '*.json'))
for f_path in tqdm(json_files, desc="Loading JSON files"):
    with open(f_path, 'r') as f:
        data = json.load(f)
        pair_name = data['pair_name']
        layer_idx = data['layer']
        if pair_name not in all_results_data:
            all_results_data[pair_name] = {}
        all_results_data[pair_name][layer_idx] = data

# 在所有循环结束后，开始绘制3D图
print("\n--- Generating 3D plots... ---")

for pair_name, layer_data in tqdm(all_results_data.items(), desc="Generating 3D Plots"):
    if not layer_data:
        continue

    # 确保层是排序的，以防万一
    sorted_layers = sorted(layer_data.keys())
    if not sorted_layers:
        continue

    num_freq_bins = len(layer_data[sorted_layers[0]]['dft_difference'])
    
    # 准备数据矩阵
    target_diff_matrix = np.zeros((num_layers, num_freq_bins))
    avg_diff_matrix = np.zeros((num_layers, num_freq_bins))

    for layer_idx in sorted_layers:
        data = layer_data[layer_idx]
        target_diff_matrix[layer_idx, :] = data['dft_difference']
        avg_diff_matrix[layer_idx, :] = data['avg_dft_difference_all_tokens']

    # 准备网格
    freq_bins = np.fft.fftfreq(hidden_size, d=1.0)[:hidden_size // 2]
    layer_indices = np.arange(num_layers)
    X, Y = np.meshgrid(freq_bins, layer_indices)

    # 绘制目标词差异的3D图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, target_diff_matrix, cmap='viridis')
    ax.set_title(f'3D Target Word Spectrum Difference\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude Difference', fontsize=12)
    filename_3d_target = f"{output_dir_images_3d}/{pair_name}_3d_target_diff.png"
    plt.savefig(filename_3d_target, dpi=300)
    plt.close(fig)

    # 绘制平均差异的3D图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, avg_diff_matrix, cmap='plasma')
    ax.set_title(f'3D Average Spectrum Difference\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude Difference', fontsize=12)
    filename_3d_avg = f"{output_dir_images_3d}/{pair_name}_3d_avg_diff.png"
    plt.savefig(filename_3d_avg, dpi=300)
    plt.close(fig)

print(f"\n3D plots saved to: {output_dir_images_3d}")
print("\n--- Experiment Finished ---")