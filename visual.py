import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_2d_spectrum_comparison(layer_data, hidden_size, output_dir):
    """
    Generates and saves an interactive 2D plot comparing spectrum differences of multiple pairs for a single layer.
    """
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']

    if not pairs_data:
        print(f"No data for layer {layer_idx} to generate interactive 2D plot.")
        return

    freq = np.fft.fftfreq(hidden_size, d=1.0)[:hidden_size // 2]

    # Define color palettes
    warm_colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    cool_colors = ['#33FF57', '#33D4FF', '#3361FF', '#8D33FF', '#D433FF']

    # --- Target Word Spectrum Difference ---
    fig_target = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_target.add_trace(go.Scatter(
            x=freq,
            y=pair_result['dft_difference'],
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_target.update_layout(
        title=f'Target Word Spectrum Difference Comparison - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude Difference',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_target = os.path.join(output_dir, f"layer_{layer_idx}_target_diff_comparison.html")
    fig_target.write_html(filename_target)

    # --- Average Spectrum Difference ---
    fig_avg = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_avg.add_trace(go.Scatter(
            x=freq,
            y=pair_result['avg_dft_difference_all_tokens'],
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_avg.update_layout(
        title=f'Average Spectrum Difference Comparison - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude Difference',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_avg = os.path.join(output_dir, f"layer_{layer_idx}_avg_diff_comparison.html")
    fig_avg.write_html(filename_avg)


def plot_sentence_A_spectrum(layer_data, hidden_size, output_dir):
    """
    Plot spectrum for sentence A across different tasks per layer
    """
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']

    if not pairs_data:
        print(f"No data for layer {layer_idx} to generate sentence A spectrum plot.")
        return

    # Define color palettes
    warm_colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    cool_colors = ['#33FF57', '#33D4FF', '#3361FF', '#8D33FF', '#D433FF']

    # --- Target Word Spectrum A ---
    fig_target = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        target_idx_A = pair_result['target_idx_A']
        dft_all_A = np.array(pair_result['dft_all_tokens_A'])
        
        # Extract target word spectrum
        dft_target_A = dft_all_A[target_idx_A]
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_target.add_trace(go.Scatter(
            x=freq,
            y=dft_target_A,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_target.update_layout(
        title=f'Sentence A Target Word Spectrum - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_target = os.path.join(output_dir, f"sentence_A_layer_{layer_idx}_target.html")
    fig_target.write_html(filename_target)

    # --- Average Spectrum A ---
    fig_avg = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        dft_all_A = np.array(pair_result['dft_all_tokens_A'])
        
        # Calculate average spectrum across all tokens
        avg_dft_A = np.mean(dft_all_A, axis=0)
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_avg.add_trace(go.Scatter(
            x=freq,
            y=avg_dft_A,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_avg.update_layout(
        title=f'Sentence A Average Spectrum - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_avg = os.path.join(output_dir, f"sentence_A_layer_{layer_idx}_avg.html")
    fig_avg.write_html(filename_avg)


def plot_sentence_B_spectrum(layer_data, hidden_size, output_dir):
    """
    Plot spectrum for sentence B across different tasks per layer
    """
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']

    if not pairs_data:
        print(f"No data for layer {layer_idx} to generate sentence B spectrum plot.")
        return

    # Define color palettes
    warm_colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    cool_colors = ['#33FF57', '#33D4FF', '#3361FF', '#8D33FF', '#D433FF']

    # --- Target Word Spectrum B ---
    fig_target = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        target_idx_B = pair_result['target_idx_B']
        dft_all_B = np.array(pair_result['dft_all_tokens_B'])
        
        # Extract target word spectrum
        dft_target_B = dft_all_B[target_idx_B]
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_target.add_trace(go.Scatter(
            x=freq,
            y=dft_target_B,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_target.update_layout(
        title=f'Sentence B Target Word Spectrum - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_target = os.path.join(output_dir, f"sentence_B_layer_{layer_idx}_target.html")
    fig_target.write_html(filename_target)

    # --- Average Spectrum B ---
    fig_avg = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        dft_all_B = np.array(pair_result['dft_all_tokens_B'])
        
        # Calculate average spectrum across all tokens
        avg_dft_B = np.mean(dft_all_B, axis=0)
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_avg.add_trace(go.Scatter(
            x=freq,
            y=avg_dft_B,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_avg.update_layout(
        title=f'Sentence B Average Spectrum - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_avg = os.path.join(output_dir, f"sentence_B_layer_{layer_idx}_avg.html")
    fig_avg.write_html(filename_avg)


def plot_diff_spectrum(layer_data, hidden_size, output_dir):
    """
    Plot spectrum difference between sentences A and B across different tasks per layer
    """
    layer_idx = layer_data['layer']
    pairs_data = layer_data['pairs']

    if not pairs_data:
        print(f"No data for layer {layer_idx} to generate diff spectrum plot.")
        return

    # Define color palettes
    warm_colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    cool_colors = ['#33FF57', '#33D4FF', '#3361FF', '#8D33FF', '#D433FF']

    # --- Target Word Spectrum Difference ---
    fig_target = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        target_idx_A = pair_result['target_idx_A']
        target_idx_B = pair_result['target_idx_B']
        dft_all_A = np.array(pair_result['dft_all_tokens_A'])
        dft_all_B = np.array(pair_result['dft_all_tokens_B'])
        
        # Calculate target word spectrum difference
        dft_target_diff = dft_all_A[target_idx_A] - dft_all_B[target_idx_B]
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_target.add_trace(go.Scatter(
            x=freq,
            y=dft_target_diff,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_target.update_layout(
        title=f'Target Word Spectrum Difference - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude Difference',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_target = os.path.join(output_dir, f"diff_layer_{layer_idx}_target.html")
    fig_target.write_html(filename_target)

    # --- Average Spectrum Difference ---
    fig_avg = go.Figure()

    syntax_color_idx = 0
    math_color_idx = 0

    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        dft_all_A = np.array(pair_result['dft_all_tokens_A'])
        dft_all_B = np.array(pair_result['dft_all_tokens_B'])
        
        # Calculate average spectrum difference
        avg_dft_A = np.mean(dft_all_A, axis=0)
        avg_dft_B = np.mean(dft_all_B, axis=0)
        avg_dft_diff = avg_dft_A - avg_dft_B
        
        if 'syntax' in pair_name:
            color = warm_colors[syntax_color_idx % len(warm_colors)]
            syntax_color_idx += 1
        else:  # math
            color = cool_colors[math_color_idx % len(cool_colors)]
            math_color_idx += 1

        fig_avg.add_trace(go.Scatter(
            x=freq,
            y=avg_dft_diff,
            mode='lines',
            name=pair_name,
            line=dict(color=color)
        ))

    fig_avg.update_layout(
        title=f'Average Spectrum Difference - Layer {layer_idx}',
        xaxis_title='Frequency',
        yaxis_title='Amplitude Difference',
        legend_title='Test Pairs',
        template='plotly_white'
    )

    filename_avg = os.path.join(output_dir, f"diff_layer_{layer_idx}_avg.html")
    fig_avg.write_html(filename_avg)


def plot_3d_spectrum_all_types(pair_name, layer_data, num_layers, hidden_size, output_dir):
    """
    Generate and save 3D spectrum plots for sentence A, sentence B, and their differences
    """
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
    target_A_matrix = np.zeros((num_layers, num_freq_bins))
    target_B_matrix = np.zeros((num_layers, num_freq_bins))
    target_diff_matrix = np.zeros((num_layers, num_freq_bins))
    avg_A_matrix = np.zeros((num_layers, num_freq_bins))
    avg_B_matrix = np.zeros((num_layers, num_freq_bins))
    avg_diff_matrix = np.zeros((num_layers, num_freq_bins))

    for layer_idx in sorted_layers:
        if layer_idx < num_layers and layer_data[layer_idx]: # Ensure layer_idx is within bounds
            data = layer_data[layer_idx]
            target_idx_A = data['target_idx_A']
            target_idx_B = data['target_idx_B']
            dft_all_A = np.array(data['dft_all_tokens_A'])
            dft_all_B = np.array(data['dft_all_tokens_B'])
            
            # Extract target word spectra
            target_A_matrix[layer_idx, :] = dft_all_A[target_idx_A]
            target_B_matrix[layer_idx, :] = dft_all_B[target_idx_B]
            target_diff_matrix[layer_idx, :] = dft_all_A[target_idx_A] - dft_all_B[target_idx_B]
            
            # Calculate average spectra
            avg_A_matrix[layer_idx, :] = np.mean(dft_all_A, axis=0)
            avg_B_matrix[layer_idx, :] = np.mean(dft_all_B, axis=0)
            avg_diff_matrix[layer_idx, :] = avg_A_matrix[layer_idx, :] - avg_B_matrix[layer_idx, :]

    # Prepare meshgrid for plotting
    layer_indices = np.arange(num_layers)
    X, Y = np.meshgrid(freq_bins, layer_indices)

    # Plot 1: 3D Sentence A Target Word Spectrum
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, target_A_matrix, cmap='viridis')
    ax.set_title(f'3D Sentence A Target Word Spectrum\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude', fontsize=12)
    
    filename_3d = os.path.join(output_dir, f"{pair_name}_sentence_A_target.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D sentence A target plot to {filename_3d}")

    # Plot 2: 3D Sentence B Target Word Spectrum
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, target_B_matrix, cmap='plasma')
    ax.set_title(f'3D Sentence B Target Word Spectrum\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude', fontsize=12)

    filename_3d = os.path.join(output_dir, f"{pair_name}_sentence_B_target.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D sentence B target plot to {filename_3d}")

    # Plot 3: 3D Target Word Spectrum Difference
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, target_diff_matrix, cmap='coolwarm')
    ax.set_title(f'3D Target Word Spectrum Difference\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude Difference', fontsize=12)
    
    filename_3d = os.path.join(output_dir, f"{pair_name}_diff_target.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D target diff plot to {filename_3d}")

    # Plot 4: 3D Sentence A Average Spectrum
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, avg_A_matrix, cmap='viridis')
    ax.set_title(f'3D Sentence A Average Spectrum\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude', fontsize=12)

    filename_3d = os.path.join(output_dir, f"{pair_name}_sentence_A_avg.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D sentence A avg plot to {filename_3d}")

    # Plot 5: 3D Sentence B Average Spectrum
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, avg_B_matrix, cmap='plasma')
    ax.set_title(f'3D Sentence B Average Spectrum\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude', fontsize=12)

    filename_3d = os.path.join(output_dir, f"{pair_name}_sentence_B_avg.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D sentence B avg plot to {filename_3d}")

    # Plot 6: 3D Average Spectrum Difference
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, avg_diff_matrix, cmap='coolwarm')
    ax.set_title(f'3D Average Spectrum Difference\n({pair_name})', fontsize=16)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Amplitude Difference', fontsize=12)

    filename_3d = os.path.join(output_dir, f"{pair_name}_diff_avg.png")
    plt.savefig(filename_3d, dpi=300)
    plt.close(fig)
    print(f"Saved 3D avg diff plot to {filename_3d}")

def generate_all_plots(all_layer_data, hidden_size, num_layers, output_dir_images):
    import glob
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