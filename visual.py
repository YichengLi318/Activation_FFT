import json
import os
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio

pio.templates.default = "plotly_white"

def generate_dft_plots(analysis_dir='output/analysis', image_dir='output/images'):
    """
    Generates and saves interactive HTML plots comparing the DFT power spectra.
    Each plot shows the top 5 differentiating dimensions for a single layer.
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
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        for i, dim_index in enumerate(top_indices):
            spec_novel = novel_spectra[i]
            spec_science = science_spectra[i]
            spec_diff = [n - s for n, s in zip(spec_novel, spec_science)]

            fig.add_trace(go.Scatter(
                x=frequencies, 
                y=spec_diff,
                mode='lines',
                name=f'Dimension {dim_index}',
                line=dict(color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Normalized Frequency',
            yaxis_title='Power Spectrum Difference (Novel - Science)',
            legend_title="Dimensions",
            hovermode="x unified"
        )
        
        output_path = os.path.join(image_dir, f'{layer_folder}_dft_comparison.html')
        fig.write_html(output_path)

    print(f"DFT plots generated and saved in {image_dir}")

if __name__ == '__main__':
    generate_dft_plots()