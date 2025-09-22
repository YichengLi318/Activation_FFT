import json
import os
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio

pio.templates.default = "plotly_white"

def generate_dft_plots(analysis_dir='output/analysis', image_dir='output/images'):
    """
    Generates and saves interactive HTML plots comparing the DFT power spectra.
    Each plot shows the top 5 differentiating dimensions for a single layer, 
    with separate curves for 'novel' and 'science' corpora.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    layer_folders = [f for f in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, f))]

    for layer_folder in tqdm(layer_folders, desc="Generating DFT Plots"):
        analysis_filepath = os.path.join(analysis_dir, layer_folder, 'dft_analysis.json')
        
        if not os.path.exists(analysis_filepath):
            print(f"Skipping {layer_folder} because dft_analysis.json not found.")
            continue

        with open(analysis_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        top_indices = data.get('top_5_diff_indices', [])
        novel_spectra = data.get('novel_spectra_top_5', [])
        science_spectra = data.get('science_spectra_top_5', [])
        frequencies = data.get('normalized_frequency_axis', [])

        if not all([top_indices, novel_spectra, science_spectra, frequencies]):
            print(f"Skipping {layer_folder} due to missing or empty data in JSON.")
            continue

        layer_number = layer_folder.split('_')[-1]
        title = f'Layer {layer_number}: Top 5 Differentiating Dimensions (DFT Power Spectrum)'
        
        fig = go.Figure()
        
        # Define warm and cool color pairs for novel and science spectra
        color_pairs = [('red', 'blue'), ('orange', 'cyan'), ('magenta', 'teal'), ('tomato', 'royalblue'), ('coral', 'deepskyblue')]

        for i, dim_index in enumerate(top_indices):
            spec_novel = novel_spectra[i]
            spec_science = science_spectra[i]
            
            warm_color, cool_color = color_pairs[i % len(color_pairs)]

            # Add trace for Novel spectrum (warm color)
            fig.add_trace(go.Scatter(
                x=frequencies, 
                y=spec_novel,
                mode='lines',
                name=f'Dimension {dim_index} (Novel)',
                legendgroup=f'dim_{dim_index}',
                line=dict(color=warm_color)
            ))

            # Add trace for Science spectrum (cool color)
            fig.add_trace(go.Scatter(
                x=frequencies, 
                y=spec_science,
                mode='lines',
                name=f'Dimension {dim_index} (Science)',
                legendgroup=f'dim_{dim_index}',
                line=dict(color=cool_color)
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Normalized Frequency',
            yaxis_title='Power Spectrum',
            legend_title="Dimensions",
            hovermode="x unified"
        )
        
        output_path = os.path.join(image_dir, f'{layer_folder}_dft_comparison.html')
        fig.write_html(output_path)

    print(f"DFT plots generated and saved in {image_dir}")

if __name__ == '__main__':
    generate_dft_plots()