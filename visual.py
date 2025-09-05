import os
import json
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ColorManager:
    def __init__(self):
        self.syntax_colors = ['red', 'orange', 'yellow']
        self.math_colors = ['violet', 'cyan', 'blue']
        self.color_map = {
            'syntax': self.syntax_colors,
            'math': self.math_colors
        }
        self.type_to_catalog = {
            'Determiner-Noun-Agreement': 'syntax',
            'Subject-Verb-Agreement': 'syntax',
            'Verb-Tense-Consistency': 'syntax',
            'Arithmetic': 'math',
            'Equation': 'math',
            'Word Problem': 'math'
        }
        self.assigned_colors = {}
        self.catalog_counters = { 'syntax': 0, 'math': 0 }

    def get_color(self, filename):
        file_type = os.path.basename(filename).split('.')[0]
        
        if file_type in self.assigned_colors:
            return self.assigned_colors[file_type]

        catalog = self.type_to_catalog.get(file_type)

        if catalog and catalog in self.color_map:
            color_list = self.color_map[catalog]
            color_index = self.catalog_counters[catalog] % len(color_list)
            color = color_list[color_index]
            
            self.assigned_colors[file_type] = color
            self.catalog_counters[catalog] += 1
            return color
        
        return 'grey' # Fallback color

class PlotlyPlotBuilder:
    """Builds a Plotly figure for spectrum analysis."""
    def __init__(self, title):
        self.fig = make_subplots(rows=1, cols=1)
        self.fig.update_layout(
            title_text=title,
            xaxis_title="Frequency",
            yaxis_title="Magnitude",
            legend_title="Task Types"
        )

    def add_trace(self, x, y, name, color):
        self.fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color)))

    def save(self, file_path):
        self.fig.write_html(file_path)
        print(f"Plot saved to {file_path}")

def load_and_process_data(data_dir):
    """Loads all JSON data from the specified directory and groups it by layer and type."""
    data_by_layer_type = {}
    layer_dirs = glob.glob(os.path.join(data_dir, 'layer_*'))
    for layer_dir in layer_dirs:
        layer_index = int(os.path.basename(layer_dir).split('_')[1])
        data_by_layer_type[layer_index] = {}
        # In the previous step, files are saved as f"{catalog}_{type}.json"
        type_files = glob.glob(os.path.join(layer_dir, '*.json'))
        for type_file in type_files:
            type_name = os.path.basename(type_file).replace('.json', '')
            with open(type_file, 'r') as f:
                data_by_layer_type[layer_index][type_name] = json.load(f)
    return data_by_layer_type

def calculate_averages(data):
    """Calculates the average DFTs for a list of data entries."""
    if not data:
        return None, None, None
    
    avg_dft_A = np.mean([item['dft_avg_A'] for item in data], axis=0)
    avg_dft_B = np.mean([item['dft_avg_B'] for item in data], axis=0)
    frequencies = data[0]['frequencies'] # Frequencies are the same for all
    return frequencies, avg_dft_A, avg_dft_B

def plot_average_spectrums(layer_index, type_data, output_dir, color_manager):
    """
    Plots the average spectrums for Sentence A, Sentence B, and their difference
    for all types within a single layer, with legend items grouped by catalog.
    """
    plot_title_A = f'Layer {layer_index}: Average Spectrum for Sentence A'
    plot_title_B = f'Layer {layer_index}: Average Spectrum for Sentence B'
    plot_title_Diff = f'Layer {layer_index}: Average Spectrum Difference (A-B)'
    
    builder_A = PlotlyPlotBuilder(plot_title_A)
    builder_B = PlotlyPlotBuilder(plot_title_B)
    builder_Diff = PlotlyPlotBuilder(plot_title_Diff)

    math_tasks = []
    syntax_tasks = []
    other_tasks = []

    for type_name, data_list in type_data.items():
        catalog = color_manager.type_to_catalog.get(type_name)
        if catalog == 'math':
            math_tasks.append((type_name, data_list))
        elif catalog == 'syntax':
            syntax_tasks.append((type_name, data_list))
        else:
            other_tasks.append((type_name, data_list))

    # Sort within each category for consistent ordering
    math_tasks.sort()
    syntax_tasks.sort()
    other_tasks.sort()

    # Combine them in the desired order: math, then syntax, then others
    sorted_tasks = math_tasks + syntax_tasks + other_tasks
    
    for type_name, data_list in sorted_tasks:
        freq, avg_A, avg_B = calculate_averages(data_list)
        if freq is not None:
            # Calculate the difference between the spectrums (without absolute value)
            avg_diff = np.array(avg_A) - np.array(avg_B)
            
            color = color_manager.get_color(type_name)
            # Use a more readable name for the legend
            trace_name = type_name.replace('_', ' ').title()

            builder_A.add_trace(freq, avg_A, trace_name, color)
            builder_B.add_trace(freq, avg_B, trace_name, color)
            builder_Diff.add_trace(freq, avg_diff, trace_name, color)

    builder_A.save(os.path.join(output_dir, f'layer_{layer_index}_avg_spectrum_A.html'))
    builder_B.save(os.path.join(output_dir, f'layer_{layer_index}_avg_spectrum_B.html'))
    builder_Diff.save(os.path.join(output_dir, f'layer_{layer_index}_avg_spectrum_Diff.html'))

def generate_all_plots(data_dir, images_dir):
    """Main function to generate all plots."""
    print("\n--- Starting Plot Generation ---")
    data = load_and_process_data(data_dir)
    color_manager = ColorManager()

    if not data:
        print("No data found to plot. Please run the analysis first.")
        return

    for layer_index, type_data in sorted(data.items()):
        plot_average_spectrums(layer_index, type_data, images_dir, color_manager)

    print("\n--- Plot generation complete. HTML files saved to output/images/ ---")

if __name__ == '__main__':
    DATA_DIR = os.path.join('output', 'data')
    IMAGES_DIR = os.path.join('output', 'images')
    os.makedirs(IMAGES_DIR, exist_ok=True)
    generate_all_plots(DATA_DIR, IMAGES_DIR)