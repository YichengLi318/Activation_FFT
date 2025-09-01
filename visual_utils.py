"""Visual utilities module for spectrum plotting.

This module contains common utilities for spectrum visualization,
including color management, data processing, and plot configuration.
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os


class ColorManager:
    """Manages color schemes for different task types."""
    
    def __init__(self):
        self.warm_colors = ['#FF5733', '#FF8C33', '#FFB833', '#FFE533', '#C7FF33']
        self.cool_colors = ['#33FF57', '#33D4FF', '#3361FF', '#8D33FF', '#D433FF']
        self.syntax_color_idx = 0
        self.math_color_idx = 0
    
    def get_color(self, pair_name):
        """Get color based on task type (syntax or math)."""
        if 'syntax' in pair_name:
            color = self.warm_colors[self.syntax_color_idx % len(self.warm_colors)]
            self.syntax_color_idx += 1
        else:  # math
            color = self.cool_colors[self.math_color_idx % len(self.cool_colors)]
            self.math_color_idx += 1
        return color
    
    def reset_indices(self):
        """Reset color indices for new plot."""
        self.syntax_color_idx = 0
        self.math_color_idx = 0


class SpectrumDataProcessor:
    """Processes spectrum data for plotting."""
    
    @staticmethod
    def extract_target_spectrum(pair_result, sentence_type='A'):
        """Extract target word spectrum from pair result."""
        if sentence_type == 'A':
            target_idx = pair_result['target_idx_A']
            dft_all = np.array(pair_result['dft_all_tokens_A'])
        else:  # sentence_type == 'B'
            target_idx = pair_result['target_idx_B']
            dft_all = np.array(pair_result['dft_all_tokens_B'])
        
        return dft_all[target_idx]
    
    @staticmethod
    def extract_average_spectrum(pair_result, sentence_type='A'):
        """Extract average spectrum from pair result."""
        if sentence_type == 'A':
            dft_all = np.array(pair_result['dft_all_tokens_A'])
        else:  # sentence_type == 'B'
            dft_all = np.array(pair_result['dft_all_tokens_B'])
        
        return np.mean(dft_all, axis=0)
    
    @staticmethod
    def calculate_spectrum_difference(pair_result, spectrum_type='target'):
        """Calculate spectrum difference between sentences A and B."""
        if spectrum_type == 'target':
            target_idx_A = pair_result['target_idx_A']
            target_idx_B = pair_result['target_idx_B']
            dft_all_A = np.array(pair_result['dft_all_tokens_A'])
            dft_all_B = np.array(pair_result['dft_all_tokens_B'])
            return dft_all_A[target_idx_A] - dft_all_B[target_idx_B]
        else:  # spectrum_type == 'average'
            dft_all_A = np.array(pair_result['dft_all_tokens_A'])
            dft_all_B = np.array(pair_result['dft_all_tokens_B'])
            avg_A = np.mean(dft_all_A, axis=0)
            avg_B = np.mean(dft_all_B, axis=0)
            return avg_A - avg_B


class PlotlyPlotBuilder:
    """Builds Plotly plots with consistent styling."""
    
    def __init__(self, title, xaxis_title='Frequency', yaxis_title='Amplitude'):
        self.fig = go.Figure()
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
    
    def add_spectrum_trace(self, freq, spectrum, name, color):
        """Add a spectrum trace to the plot."""
        self.fig.add_trace(go.Scatter(
            x=freq,
            y=spectrum,
            mode='lines',
            name=name,
            line=dict(color=color)
        ))
    
    def finalize_layout(self):
        """Apply final layout settings."""
        self.fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_title,
            yaxis_title=self.yaxis_title,
            legend_title='Test Pairs',
            template='plotly_white'
        )
    
    def save_html(self, filename):
        """Save plot as HTML file."""
        self.fig.write_html(filename)


class Matplotlib3DPlotBuilder:
    """Builds 3D matplotlib plots with consistent styling."""
    
    def __init__(self, title, figsize=(14, 10)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.title = title
    
    def plot_surface(self, X, Y, Z, cmap='viridis'):
        """Plot 3D surface."""
        self.ax.plot_surface(X, Y, Z, cmap=cmap)
    
    def set_labels(self, xlabel='Frequency', ylabel='Layer', zlabel='Amplitude'):
        """Set axis labels."""
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_zlabel(zlabel, fontsize=12)
    
    def set_title(self, subtitle=''):
        """Set plot title."""
        full_title = f'{self.title}\n({subtitle})' if subtitle else self.title
        self.ax.set_title(full_title, fontsize=16)
    
    def save_and_close(self, filename, dpi=300):
        """Save plot and close figure."""
        plt.savefig(filename, dpi=dpi)
        plt.close(self.fig)
        print(f"Saved plot to {filename}")


def create_2d_spectrum_plots(pairs_data, layer_idx, output_dir, 
                            spectrum_type='target', sentence_type='A',
                            plot_type='comparison'):
    """Generic function to create 2D spectrum plots.
    
    Args:
        pairs_data: List of pair results
        layer_idx: Layer index
        output_dir: Output directory
        spectrum_type: 'target' or 'average'
        sentence_type: 'A', 'B', or 'diff'
        plot_type: 'comparison' or 'individual'
    """
    color_manager = ColorManager()
    processor = SpectrumDataProcessor()
    
    # Determine plot title and filename
    if sentence_type == 'diff':
        if spectrum_type == 'target':
            title = f'Target Word Spectrum Difference - Layer {layer_idx}'
            filename = f"diff_layer_{layer_idx}_target.html"
            yaxis_title = 'Amplitude Difference'
        else:
            title = f'Average Spectrum Difference - Layer {layer_idx}'
            filename = f"diff_layer_{layer_idx}_avg.html"
            yaxis_title = 'Amplitude Difference'
    else:
        if spectrum_type == 'target':
            title = f'Sentence {sentence_type} Target Word Spectrum - Layer {layer_idx}'
            filename = f"sentence_{sentence_type}_layer_{layer_idx}_target.html"
        else:
            title = f'Sentence {sentence_type} Average Spectrum - Layer {layer_idx}'
            filename = f"sentence_{sentence_type}_layer_{layer_idx}_avg.html"
        yaxis_title = 'Amplitude'
    
    plot_builder = PlotlyPlotBuilder(title, yaxis_title=yaxis_title)
    
    # Add traces for each pair
    for pair_result in pairs_data:
        pair_name = pair_result['pair_name']
        freq = pair_result['frequencies']
        color = color_manager.get_color(pair_name)
        
        # Extract spectrum data based on type
        if sentence_type == 'diff':
            spectrum = processor.calculate_spectrum_difference(pair_result, spectrum_type)
        else:
            if spectrum_type == 'target':
                spectrum = processor.extract_target_spectrum(pair_result, sentence_type)
            else:
                spectrum = processor.extract_average_spectrum(pair_result, sentence_type)
        
        plot_builder.add_spectrum_trace(freq, spectrum, pair_name, color)
    
    plot_builder.finalize_layout()
    full_filename = os.path.join(output_dir, filename)
    plot_builder.save_html(full_filename)


def create_3d_spectrum_plot(X, Y, Z_matrix, pair_name, plot_type, output_dir, 
                           cmap='viridis', zlabel='Amplitude'):
    """Generic function to create 3D spectrum plots.
    
    Args:
        X, Y: Meshgrid coordinates
        Z_matrix: Data matrix for plotting
        pair_name: Name of the pair
        plot_type: Type of plot (e.g., 'sentence_A_target')
        output_dir: Output directory
        cmap: Colormap
        zlabel: Z-axis label
    """
    title_map = {
        'sentence_A_target': '3D Sentence A Target Word Spectrum',
        'sentence_B_target': '3D Sentence B Target Word Spectrum',
        'sentence_A_avg': '3D Sentence A Average Spectrum',
        'sentence_B_avg': '3D Sentence B Average Spectrum',
        'diff_target': '3D Target Word Spectrum Difference',
        'diff_avg': '3D Average Spectrum Difference'
    }
    
    title = title_map.get(plot_type, '3D Spectrum Plot')
    plot_builder = Matplotlib3DPlotBuilder(title)
    
    plot_builder.plot_surface(X, Y, Z_matrix, cmap=cmap)
    plot_builder.set_labels(zlabel=zlabel)
    plot_builder.set_title(pair_name)
    
    filename = os.path.join(output_dir, f"{pair_name}_{plot_type}.png")
    plot_builder.save_and_close(filename)