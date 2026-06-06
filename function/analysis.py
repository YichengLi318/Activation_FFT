import json
import os
import numpy as np
from tqdm import tqdm
import glob
import re


def analyze_dft_difference(input_dir, output_dir, selected_dims):
    """
    Performs DFT analysis per layer for each text category.
    - Supports input structure: output/data/<dataset>/<category>/layer_<idx>/sample_*.json
    - Saves results under output/analysis/<dataset>/layer_<idx>/dft_analysis_<category>_<dataset>_layer<idx>.json
    - Uses actual frequency values (cycles/token) without interpolation.
    - If selected_dims is None, randomly pick up to 20 dims from first sample.
    - Also stores amplitude and phase so later frequency-domain inversion is possible.
    """
    dataset_name = os.path.basename(os.path.normpath(input_dir))

    os.makedirs(output_dir, exist_ok=True)

    # Categories under input_dir
    categories = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if not categories:
        print(f"Warning: No categories found in {input_dir}.")
        return

    # Collect available layers by scanning the first category
    first_cat_layers = sorted([d for d in os.listdir(os.path.join(input_dir, categories[0])) if d.startswith('layer_') and os.path.isdir(os.path.join(input_dir, categories[0], d))])
    if not first_cat_layers:
        print(f"Warning: No layer directories found under category '{categories[0]}' in {input_dir}.")
        return

    for layer_dir in tqdm(first_cat_layers, desc="Analyzing Layers Individually"):
        m = re.search(r"layer[ _](\d+)", layer_dir)
        layer_number = m.group(1) if m else layer_dir
        analysis_layer_path = os.path.join(output_dir, f"layer_{layer_number}")
        os.makedirs(analysis_layer_path, exist_ok=True)

        for category in tqdm(categories, desc=f"Layer {layer_dir} - Processing Categories", leave=False):
            category_path = os.path.join(input_dir, category, layer_dir)
            if not os.path.isdir(category_path):
                continue
            activation_files = glob.glob(os.path.join(category_path, '*.json'))
            if not activation_files:
                print(f"Warning: No activation files found in {category_path}. Skipping category.")
                continue

            power_list = []
            valid_selected = None
            freq_axis = None
            seq_len_meta = None
            per_sample_amp_phase = []

            for file_path in activation_files:
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        activations = np.array(data['activations'])
                    except (json.JSONDecodeError, KeyError):
                        print(f"Warning: Could not process file {file_path}. Skipping.")
                        continue

                if activations.ndim < 2 or activations.shape[0] < 2 or activations.shape[1] == 0:
                    continue

                activations = activations.T  # dims x seq_len
                activations = activations - np.mean(activations, axis=1, keepdims=True)
                window = np.hanning(activations.shape[1])

                # Randomly select dims if not provided (use first file's shape)
                if valid_selected is None:
                    if selected_dims and len(selected_dims) > 0:
                        valid_selected = [d for d in selected_dims if 0 <= d < activations.shape[0]]
                    else:
                        rng = np.random.default_rng()
                        valid_selected = rng.choice(activations.shape[0], size=min(20, activations.shape[0]), replace=False).tolist()
                    if len(valid_selected) == 0:
                        continue

                sel_acts = activations[valid_selected, :]
                windowed_activations = sel_acts * window

                # Use one-sided real FFT and absolute frequency axis derived from padded token length
                num_samples = sel_acts.shape[1]
                rfft_result = np.fft.rfft(windowed_activations, axis=1)
                amplitude = np.abs(rfft_result)
                phase = np.angle(rfft_result)
                power_spectrum = amplitude**2
                current_freq_axis = np.fft.rfftfreq(num_samples, d=1)

                if freq_axis is None:
                    freq_axis = current_freq_axis
                    seq_len_meta = int(num_samples)
                power_list.append(power_spectrum)
                per_sample_amp_phase.append({
                    "file": os.path.basename(file_path),
                    "amplitude": amplitude.tolist(),
                    "phase": phase.tolist(),
                })

            if not power_list:
                print(f"Warning: No valid spectra generated for category {category} in layer {layer_dir}.")
                continue

            all_power = np.array(power_list)  # texts x dims x freqs
            mean_power = np.mean(all_power, axis=0)
            std_power = np.std(all_power, axis=0)

            amplitude_array = np.array([item["amplitude"] for item in per_sample_amp_phase])
            phase_array = np.array([item["phase"] for item in per_sample_amp_phase])
            mean_amplitude = np.mean(amplitude_array, axis=0)
            std_amplitude = np.std(amplitude_array, axis=0)

            category_label = f"{category}_{dataset_name}"
            output_data = {
                "category": category_label,
                "selected_indices": valid_selected if valid_selected is not None else (selected_dims or []),
                "spectra_selected": mean_power.tolist(),  # Legacy field: mean power spectrum
                "std_selected": std_power.tolist(),       # Legacy field: power spectrum stddev
                "amplitude_mean": mean_amplitude.tolist(),
                "amplitude_std": std_amplitude.tolist(),
                "samples": per_sample_amp_phase,          # Per-sample amplitude and phase
                "frequency_axis": (freq_axis.tolist() if freq_axis is not None else []),
                "sequence_length": seq_len_meta if seq_len_meta is not None else 0,
            }

            output_filename = f'dft_analysis_{category}_{dataset_name}_layer{layer_number}.json'
            output_file = os.path.join(analysis_layer_path, output_filename)
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)

    print(f"Individual DFT analysis completed for dataset '{dataset_name}'.")
