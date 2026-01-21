"""
Plotting utilities for neutron data visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def initialize_plots(ax1, ax2, ax3, ax4, font_sizes):
    """Initialize all plots with proper formatting"""
    # Set titles and labels with better font sizes
    ax1.set_title("2D Color Map - S(Q,ω)", fontsize=font_sizes['title'])
    ax1.set_xlabel("ω (meV)", fontsize=font_sizes['label'])
    ax1.set_ylabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
    ax1.tick_params(axis='both', labelsize=font_sizes['tick'])
    
    ax2.set_title("Current Spectrum", fontsize=font_sizes['title'])
    ax2.set_xlabel("ω (meV)", fontsize=font_sizes['label'])
    ax2.set_ylabel("S(ω)", fontsize=font_sizes['label'])
    ax2.tick_params(axis='both', labelsize=font_sizes['tick'])
    
    ax3.set_title("Dispersion Relation", fontsize=font_sizes['title'])
    ax3.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
    ax3.set_ylabel("Peak Center (meV)", fontsize=font_sizes['label'])
    ax3.tick_params(axis='both', labelsize=font_sizes['tick'])
    
    ax4.set_title("Parameter Trends", fontsize=font_sizes['title'])
    ax4.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
    ax4.set_ylabel("Parameter Value", fontsize=font_sizes['label'])
    ax4.tick_params(axis='both', labelsize=font_sizes['tick'])
    
    # Add grid for better readability
    for ax in [ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3, linestyle='--')

def update_plots(ax1, ax2, ax3, ax4, data, current_q_index, fit_results, font_sizes, canvas, fig):
    """Update all plots with current data"""
    # Clear plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.clear()
    
    # Plot 1: 2D color map
    if len(data['q']) > 1 and len(data['omega']) > 1:
        im = ax1.imshow(data['S_data'], 
                       aspect='auto',
                       origin='lower',
                       extent=[data['omega'][0], data['omega'][-1], 
                              data['q'][0], data['q'][-1]],
                       cmap='viridis',
                       interpolation='nearest')
        ax1.set_title("2D Color Map - S(Q,ω)", fontsize=font_sizes['title'])
        ax1.set_xlabel("ω (meV)", fontsize=font_sizes['label'])
        ax1.set_ylabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
        ax1.tick_params(axis='both', labelsize=font_sizes['tick'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.ax.tick_params(labelsize=font_sizes['tick'])
        
        # Highlight current Q
        if current_q_index < len(data['q']):
            current_q = data['q'][current_q_index]
            ax1.axhline(y=current_q, color='red', linestyle='--', linewidth=1, alpha=0.7)
    else:
        ax1.text(0.5, 0.5, "Insufficient data\nfor 2D plot", 
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=font_sizes['title'])
    
    # Plot 2: Current spectrum
    if current_q_index < len(data['q']):
        q_val = data['q'][current_q_index]
        x_data = data['omega']
        y_data = data['S_data'][current_q_index, :]
        
        # Check if errors are available
        if 'S_errors' in data and data['S_errors'].shape == data['S_data'].shape:
            y_errors = data['S_errors'][current_q_index, :]
            ax2.errorbar(x_data, y_data, yerr=y_errors,
                        fmt='o', markersize=4, alpha=0.7, 
                        label=f'Q = {q_val:.3f} Å⁻¹',
                        capsize=2, elinewidth=1)
        else:
            ax2.plot(x_data, y_data,
                    'o-', markersize=4, alpha=0.7, 
                    label=f'Q = {q_val:.3f} Å⁻¹', linewidth=1)
        
        ax2.set_title(f"Spectrum at Q = {q_val:.3f} Å⁻¹", fontsize=font_sizes['title'])
        ax2.set_xlabel("ω (meV)", fontsize=font_sizes['label'])
        ax2.set_ylabel("S(ω)", fontsize=font_sizes['label'])
        ax2.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax2.legend(fontsize=font_sizes['legend'])
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3 & 4: Only if we have fit results
    if len(fit_results) > 0:
        plot_fit_results(ax3, ax4, data, fit_results, font_sizes, canvas)
    else:
        # Clear fit result plots
        ax3.set_title("Dispersion Relation", fontsize=font_sizes['title'])
        ax3.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
        ax3.set_ylabel("Peak Center (meV)", fontsize=font_sizes['label'])
        ax3.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        ax4.set_title("Parameter Trends", fontsize=font_sizes['title'])
        ax4.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
        ax4.set_ylabel("Parameter Value", fontsize=font_sizes['label'])
        ax4.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    fig.tight_layout()
    canvas.draw()

def plot_fit_results(ax3, ax4, data, fit_results, font_sizes, canvas):
    """Plot fit results (dispersion and parameter trends)"""
    if len(fit_results) == 0:
        return
    
    # Extract valid results
    valid_indices = []
    centers = []
    center_errors = []
    amplitudes = []
    amplitude_errors = []
    widths = []
    width_errors = []
    
    for i, result in enumerate(fit_results):
        if result is not None and len(result[0]) >= 3:
            valid_indices.append(i)
            popt, perr = result
            centers.append(popt[1])  # Center parameter
            center_errors.append(perr[1] if len(perr) > 1 else 0)
            amplitudes.append(popt[0])  # Amplitude parameter
            amplitude_errors.append(perr[0] if len(perr) > 0 else 0)
            if len(popt) > 2:
                widths.append(popt[2])  # Width parameter
                width_errors.append(perr[2] if len(perr) > 2 else 0)
    
    if not valid_indices:
        return
    
    # Plot dispersion relation
    ax3.clear()
    if len(valid_indices) > 0:
        ax3.errorbar(data['q'][valid_indices], centers, yerr=center_errors,
                    fmt='o-', linewidth=2, markersize=4, capsize=3)
        ax3.set_title("Dispersion Relation", fontsize=font_sizes['title'])
        ax3.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
        ax3.set_ylabel("Peak Center (meV)", fontsize=font_sizes['label'])
        ax3.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot parameter trends
    ax4.clear()
    if len(valid_indices) > 0:
        ax4.errorbar(data['q'][valid_indices], amplitudes, yerr=amplitude_errors,
                    fmt='o-', linewidth=2, markersize=4, capsize=3, label='Amplitude')
        
        if widths:
            ax4.errorbar(data['q'][valid_indices], widths, yerr=width_errors,
                        fmt='s-', linewidth=2, markersize=4, capsize=3, label='Width')
        
        ax4.set_title("Parameter Trends", fontsize=font_sizes['title'])
        ax4.set_xlabel("Q (Å⁻¹)", fontsize=font_sizes['label'])
        ax4.set_ylabel("Parameter Value", fontsize=font_sizes['label'])
        ax4.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax4.legend(fontsize=font_sizes['legend'])
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    canvas.draw()