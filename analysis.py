import json
import os
import numpy as np
from tqdm import tqdm
import glob

def analyze_dft_difference(input_dir, output_dir):
    """
    Analyzes the DFT difference between two categories of text data for each layer.
    This version streams data from individual JSON files, calculates FFT for each, 
    interpolates to a common frequency axis, and then averages the spectra.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    common_freq_axis = np.linspace(0, 0.5, 512)

    layer_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for layer_folder in tqdm(layer_folders, desc="Analyzing Layers"):
        layer_path = os.path.join(input_dir, layer_folder)
        analysis_layer_path = os.path.join(output_dir, layer_folder)
        os.makedirs(analysis_layer_path, exist_ok=True)

        final_spectra = {}

        for category in ['novel', 'science']:
            category_path = os.path.join(layer_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} not found for layer {layer_folder}. Skipping category.")
                continue

            activation_files = glob.glob(os.path.join(category_path, '*.json'))
            if not activation_files:
                print(f"Warning: No activation files found in {category_path}. Skipping category.")
                continue

            interpolated_spectra_list = []
            
            for file_path in tqdm(activation_files, desc=f"Processing {category}", leave=False):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        activations = np.array(data['activations'])
                    except (json.JSONDecodeError, KeyError):
                        print(f"Warning: Could not process file {file_path}. Skipping.")
                        continue

                if activations.ndim < 2 or activations.shape[0] < 2 or activations.shape[1] == 0:
                    continue
                
                activations = activations.T
                
                sum_abs_activations = np.sum(np.abs(activations), axis=1, keepdims=True)
                sum_abs_activations[sum_abs_activations == 0] = 1
                normalized_activations = activations / sum_abs_activations

                normalized_activations = normalized_activations - np.mean(normalized_activations, axis=1, keepdims=True)
                fft_result = np.fft.fft(normalized_activations, axis=1)
                power_spectrum = np.abs(fft_result)**2

                num_samples = normalized_activations.shape[1]
                original_freq_axis = np.fft.fftfreq(num_samples, d=1)[:num_samples // 2]
                
                one_sided_power_spectrum = power_spectrum[:, :num_samples // 2]

                if len(original_freq_axis) == 0:
                    continue

                interpolated_spectrum_for_text = np.array([
                    np.interp(common_freq_axis, original_freq_axis, dim_spectrum)
                    for dim_spectrum in one_sided_power_spectrum
                ])
                interpolated_spectra_list.append(interpolated_spectrum_for_text)

            if interpolated_spectra_list:
                avg_power_spectrum = np.mean(interpolated_spectra_list, axis=0)
                final_spectra[category] = avg_power_spectrum
            else:
                print(f"Warning: No valid data to process for category '{category}' in layer {layer_folder}.")

        if 'novel' not in final_spectra or 'science' not in final_spectra:
            print(f"Warning: Skipping layer {layer_folder} due to missing data for comparison.")
            continue

        novel_spectrum = final_spectra['novel']
        science_spectrum = final_spectra['science']
        
        mse = np.mean((novel_spectrum - science_spectrum)**2, axis=1)
        
        top_5_diff_indices = np.argsort(mse)[-5:][::-1]

        analysis_result = {
            'top_5_diff_indices': top_5_diff_indices.tolist(),
            'novel_spectra_top_5': novel_spectrum[top_5_diff_indices].tolist(),
            'science_spectra_top_5': science_spectrum[top_5_diff_indices].tolist(),
            'normalized_frequency_axis': common_freq_axis.tolist()
        }

        output_file = os.path.join(analysis_layer_path, 'dft_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(analysis_result, f, indent=4)

    print(f"DFT difference analysis completed for all layers.")