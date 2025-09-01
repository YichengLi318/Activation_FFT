"""Spectrum visualization module.

This module provides functions for generating 2D and 3D spectrum plots
using the visual_utils module for common functionality.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from visual_utils import (
    create_2d_spectrum_plots,
    create_3d_spectrum_plot,
    SpectrumDataProcessor
)


def plot_2d_spectrum_comparison(layer_data, hidden_size, output_dir):
    """Generate 2D spectrum comparison plots for a specific layer."""
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']
    
    # This function maintains the original comparison plot functionality
    # but could be refactored further if needed
    from visual_utils import ColorManager, PlotlyPlotBuilder
    
    color_manager = ColorManager()
    
    # Create comparison plot
    plot_builder = PlotlyPlotBuilder(
        f'2D Spectrum Comparison - Layer {layer_idx}',
        yaxis_title='Amplitude Difference'
    )
    
    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        color = color_manager.get_color(pair_name)
        
        # Calculate spectrum difference for comparison
        processor = SpectrumDataProcessor()
        spectrum_diff = processor.calculate_spectrum_difference(pair_result, 'target')
        
        plot_builder.add_spectrum_trace(freq, spectrum_diff, pair_name, color)
    
    plot_builder.finalize_layout()
    filename = os.path.join(output_dir, f"comparison_layer_{layer_idx}.html")
    plot_builder.save_html(filename)


def plot_sentence_A_spectrum(layer_data, hidden_size, output_dir):
    """Generate sentence A spectrum plots (target and average)."""
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']
    
    # Target word spectrum
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='target', sentence_type='A'
    )
    
    # Average spectrum
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='average', sentence_type='A'
    )


def plot_sentence_B_spectrum(layer_data, hidden_size, output_dir):
    """Generate sentence B spectrum plots (target and average)."""
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']
    
    # Target word spectrum
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='target', sentence_type='B'
    )
    
    # Average spectrum
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='average', sentence_type='B'
    )


def plot_diff_spectrum(layer_data, hidden_size, output_dir):
    """Generate spectrum difference plots (target and average)."""
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']
    
    # Target word spectrum difference
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='target', sentence_type='diff'
    )
    
    # Average spectrum difference
    create_2d_spectrum_plots(
        pairs_data, layer_idx, output_dir,
        spectrum_type='average', sentence_type='diff'
    )


def plot_3d_spectrum_all_types(pair_name, layer_data, num_layers, hidden_size, output_dir):
    """Generate and save 3D spectrum plots for sentence A, sentence B, and their differences."""
    if not layer_data:
        print(f"No data available for {pair_name} to generate 3D plot.")
        return

    sorted_layers = sorted(layer_data.keys())
    if not sorted_layers:
        print(f"No layers found for {pair_name}.")
        return

    # Get frequency bins from the first available layer
    first_layer_data = None
    for layer_idx in sorted_layers:
        if layer_data[layer_idx]:
            first_layer_data = layer_data[layer_idx]
            break
    
    if not first_layer_data:
        print(f"No valid data found for {pair_name}.")
        return
        
    freq_bins = first_layer_data['frequencies']
    num_freq_bins = len(freq_bins)

    # Prepare data matrices for all types
    matrices = {
        'target_A': np.zeros((num_layers, num_freq_bins)),
        'target_B': np.zeros((num_layers, num_freq_bins)),
        'target_diff': np.zeros((num_layers, num_freq_bins)),
        'avg_A': np.zeros((num_layers, num_freq_bins)),
        'avg_B': np.zeros((num_layers, num_freq_bins)),
        'avg_diff': np.zeros((num_layers, num_freq_bins))
    }

    processor = SpectrumDataProcessor()
    
    for layer_idx in sorted_layers:
        if layer_idx < num_layers and layer_data[layer_idx]:
            data = layer_data[layer_idx]
            
            # Extract spectra using the processor
            matrices['target_A'][layer_idx, :] = processor.extract_target_spectrum(data, 'A')
            matrices['target_B'][layer_idx, :] = processor.extract_target_spectrum(data, 'B')
            matrices['target_diff'][layer_idx, :] = processor.calculate_spectrum_difference(data, 'target')
            matrices['avg_A'][layer_idx, :] = processor.extract_average_spectrum(data, 'A')
            matrices['avg_B'][layer_idx, :] = processor.extract_average_spectrum(data, 'B')
            matrices['avg_diff'][layer_idx, :] = processor.calculate_spectrum_difference(data, 'average')

    # Prepare meshgrid for plotting
    layer_indices = np.arange(num_layers)
    X, Y = np.meshgrid(freq_bins, layer_indices)

    # Generate all 6 plots using the generic function
    plot_configs = [
        ('sentence_A_target', matrices['target_A'], 'viridis', 'Amplitude'),
        ('sentence_B_target', matrices['target_B'], 'plasma', 'Amplitude'),
        ('diff_target', matrices['target_diff'], 'coolwarm', 'Amplitude Difference'),
        ('sentence_A_avg', matrices['avg_A'], 'viridis', 'Amplitude'),
        ('sentence_B_avg', matrices['avg_B'], 'plasma', 'Amplitude'),
        ('diff_avg', matrices['avg_diff'], 'coolwarm', 'Amplitude Difference')
    ]
    
    for plot_type, matrix, cmap, zlabel in plot_configs:
        create_3d_spectrum_plot(X, Y, matrix, pair_name, plot_type, output_dir, cmap, zlabel)


def generate_all_plots(all_layer_data, hidden_size, num_layers, output_dir_images):
    """Generate all 2D and 3D plots."""
    from tqdm import tqdm
    
    # --- 2D Plotting ---
    print("\n--- Generating 2D plots... ---")
    for layer_data in tqdm(all_layer_data, desc="Generating 2D Sentence A Plots"):
        plot_sentence_A_spectrum(layer_data, hidden_size, output_dir_images)
    
    for layer_data in tqdm(all_layer_data, desc="Generating 2D Sentence B Plots"):
        plot_sentence_B_spectrum(layer_data, hidden_size, output_dir_images)
    
    for layer_data in tqdm(all_layer_data, desc="Generating 2D Diff Plots"):
        plot_diff_spectrum(layer_data, hidden_size, output_dir_images)
    
    print(f"\n2D plots saved to: {output_dir_images}")

    # --- 3D Plotting ---
    # Reorganize data for 3D plotting (by pair)
    all_results_data_by_pair = {}
    for layer_data in all_layer_data:
        layer_idx = layer_data['layer']
        for pair_result in layer_data['pairs']:
            pair_name = pair_result['pair_name']
            if pair_name not in all_results_data_by_pair:
                all_results_data_by_pair[pair_name] = {}
            # Use the new data structure for the 3D plot function
            all_results_data_by_pair[pair_name][layer_idx] = {
                'pair_name': pair_name,
                'layer': layer_idx,
                'frequencies': pair_result['frequencies'],
                'target_idx_A': pair_result['target_idx_A'],
                'target_idx_B': pair_result['target_idx_B'],
                'dft_all_tokens_A': pair_result['dft_all_tokens_A'],
                'dft_all_tokens_B': pair_result['dft_all_tokens_B']
            }

    print("\n--- Generating 3D plots... ---")
    for pair_name, layer_data in tqdm(all_results_data_by_pair.items(), desc="Generating 3D Plots"):
        plot_3d_spectrum_all_types(pair_name, layer_data, num_layers, hidden_size, output_dir_images)

    print(f"\n3D plots saved to: {output_dir_images}")
    print("\n--- Experiment Finished ---")