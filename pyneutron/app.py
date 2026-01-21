"""
Main GUI application for PyNeutron
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import traceback

from .data_loader import load_nexus_file, create_sample_data
from .fitting import lorentzian, double_lorentzian, gaussian, fit_spectrum
from .plotting import initialize_plots, update_plots, plot_fit_results
from .utils import export_results

class NeutronAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyNeutron - Neutron Data Analysis")
        self.root.geometry("1200x800")
        
        # Set better font sizes
        self.font_sizes = {
            'title': 12,
            'label': 10,
            'info': 9,
            'tick': 9,
            'legend': 9
        }
        
        # Initialize with sample data structure
        self.data = None
        self.current_q_index = 0
        self.fit_results = []
        
        # Create sample data if needed (for testing)
        self.create_sample_data()
        
        self.setup_ui()
        
    def create_sample_data(self):
        """Create sample data for testing when no file is loaded"""
        self.data = create_sample_data()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Right panel - Plots
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(left_frame)
        self.setup_plots(right_frame)
    
    def setup_controls(self, parent):
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load NeXus/HDF5 File", 
                  command=self.load_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Use Sample Data", 
                  command=self.use_sample_data).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=(5, 0))
        
        # Current file label
        self.file_label = ttk.Label(file_frame, text="No file loaded", wraplength=300)
        self.file_label.pack(fill=tk.X, pady=(5, 0))
        
        # Data info
        info_frame = ttk.LabelFrame(parent, text="Data Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=6, width=35, font=('Arial', 9))
        self.info_text.pack(fill=tk.X)
        
        # Q selection
        q_frame = ttk.LabelFrame(parent, text="Q Selection", padding=10)
        q_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(q_frame, text="Q Index:").pack(anchor=tk.W)
        self.q_slider = ttk.Scale(q_frame, from_=0, to=100, 
                                 orient=tk.HORIZONTAL, command=self.on_q_change)
        self.q_slider.pack(fill=tk.X)
        
        self.q_label = ttk.Label(q_frame, text="Q = 0.000 Å⁻¹", font=('Arial', 9, 'bold'))
        self.q_label.pack(anchor=tk.W)
        
        # Fitting controls
        fit_frame = ttk.LabelFrame(parent, text="Fitting Controls", padding=10)
        fit_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(fit_frame, text="Fit Current Spectrum", 
                  command=self.fit_current).pack(fill=tk.X, pady=2)
        ttk.Button(fit_frame, text="Fit All Spectra", 
                  command=self.fit_all).pack(fill=tk.X, pady=2)
        
        # Model selection
        model_frame = ttk.LabelFrame(parent, text="Fitting Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar(value="Lorentzian")
        ttk.Radiobutton(model_frame, text="Single Lorentzian", 
                       variable=self.model_var, value="Lorentzian").pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="Double Lorentzian", 
                       variable=self.model_var, value="Double Lorentzian").pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="Gaussian", 
                       variable=self.model_var, value="Gaussian").pack(anchor=tk.W)
        
        # Initial parameters
        param_frame = ttk.LabelFrame(parent, text="Initial Parameters", padding=10)
        param_frame.pack(fill=tk.X)
        
        # Make parameter entries larger
        ttk.Label(param_frame, text="Amplitude:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.amp_entry = ttk.Entry(param_frame, width=20)
        self.amp_entry.insert(0, "5.0")
        self.amp_entry.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))
        
        ttk.Label(param_frame, text="Center (meV):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.center_entry = ttk.Entry(param_frame, width=20)
        self.center_entry.insert(0, "0.0")
        self.center_entry.grid(row=1, column=1, sticky=tk.EW, padx=(5, 0))
        
        ttk.Label(param_frame, text="Width (meV):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.width_entry = ttk.Entry(param_frame, width=20)
        self.width_entry.insert(0, "1.0")
        self.width_entry.grid(row=2, column=1, sticky=tk.EW, padx=(5, 0))
        
        ttk.Label(param_frame, text="Background:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.bg_entry = ttk.Entry(param_frame, width=20)
        self.bg_entry.insert(0, "0.1")
        self.bg_entry.grid(row=3, column=1, sticky=tk.EW, padx=(5, 0))
        
        param_frame.columnconfigure(1, weight=1)
    
    def setup_plots(self, parent):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Create matplotlib figure with 2x2 subplots
        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Create subplots
        self.ax1 = plt.subplot(2, 2, 1)  # 2D Map
        self.ax2 = plt.subplot(2, 2, 2)  # Current spectrum
        self.ax3 = plt.subplot(2, 2, 3)  # Dispersion
        self.ax4 = plt.subplot(2, 2, 4)  # Parameters
        
        # Adjust layout
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, 
                           wspace=0.25, hspace=0.35)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        initialize_plots(self.ax1, self.ax2, self.ax3, self.ax4, self.font_sizes)
        self.canvas.draw()
    
    def use_sample_data(self):
        """Use the sample data for testing"""
        self.create_sample_data()
        self.update_info()
        self.update_plots()
        self.file_label.config(text="Using: Sample Data")
        messagebox.showinfo("Sample Data", "Using sample data. You can now test the fitting functions.")
    
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select NeXus/HDF5 file",
            filetypes=[("NeXus files", "*.nxs"), ("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.data = load_nexus_file(filename)
            
            # Update UI
            num_q_points = len(self.data['q'])
            if num_q_points > 1:
                self.q_slider.config(from_=0, to=num_q_points-1)
            else:
                self.q_slider.config(from_=0, to=1)
            
            # Reset fit results
            self.fit_results = []
            
            self.update_info()
            self.update_plots()
            self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}")
            messagebox.showinfo("Success", f"Successfully loaded file:\n{filename}")
            
        except Exception as e:
            error_msg = f"Failed to load file:\n{str(e)}\n\nUsing sample data instead."
            messagebox.showerror("Error", error_msg)
            print("Detailed error traceback:")
            traceback.print_exc()
            self.use_sample_data()
    
    def update_info(self):
        if self.data is not None:
            info = f"File: {self.data.get('filename', 'Unknown')}\n"
            info += f"Q range: {self.data['q'][0]:.3f} - {self.data['q'][-1]:.3f} Å⁻¹\n"
            info += f"ω range: {self.data['omega'][0]:.3f} - {self.data['omega'][-1]:.3f} meV\n"
            info += f"Data shape: {self.data['S_data'].shape}\n"
            info += f"Points: {len(self.data['q'])} Q × {len(self.data['omega'])} ω\n"
            info += f"Fits performed: {len([r for r in self.fit_results if r is not None])}"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
    
    def on_q_change(self, value):
        if self.data is not None:
            self.current_q_index = int(float(value))
            if self.current_q_index >= len(self.data['q']):
                self.current_q_index = len(self.data['q']) - 1
            self.update_plots()
    
    def update_plots(self):
        if self.data is None:
            return
            
        # Get fitting function based on selected model
        model = self.model_var.get()
        if model == "Lorentzian":
            fit_func = lorentzian
        elif model == "Double Lorentzian":
            fit_func = double_lorentzian
        else:  # Gaussian
            fit_func = gaussian
        
        # Update plots
        update_plots(self.ax1, self.ax2, self.ax3, self.ax4, 
                    self.data, self.current_q_index, 
                    self.fit_results, self.font_sizes, self.canvas, self.fig)
        
        # Update Q label
        if self.current_q_index < len(self.data['q']):
            self.q_label.config(text=f"Q = {self.data['q'][self.current_q_index]:.3f} Å⁻¹ (Index: {self.current_q_index})")
    
    def fit_current(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            # Get current spectrum
            q_val = self.data['q'][self.current_q_index]
            x_data = self.data['omega']
            y_data = self.data['S_data'][self.current_q_index, :]
            
            # Get errors if available
            if 'S_errors' in self.data and self.data['S_errors'].shape == self.data['S_data'].shape:
                y_errors = self.data['S_errors'][self.current_q_index, :]
            else:
                y_errors = np.ones_like(y_data) * 0.1
            
            # Get model and initial parameters
            model = self.model_var.get()
            
            if model == "Lorentzian":
                fit_func = lorentzian
                A0 = float(self.amp_entry.get())
                x00 = float(self.center_entry.get())
                gamma0 = float(self.width_entry.get())
                C0 = float(self.bg_entry.get())
                p0 = [A0, x00, gamma0, C0]
                bounds = ([0, -10, 0.1, 0], [np.inf, 10, 5, np.inf])
                
            elif model == "Double Lorentzian":
                fit_func = double_lorentzian
                A0 = float(self.amp_entry.get())
                x00 = float(self.center_entry.get())
                gamma0 = float(self.width_entry.get())
                C0 = float(self.bg_entry.get())
                # For double Lorentzian, use symmetric initial guess
                p0 = [A0, x00, gamma0, A0/2, -x00, gamma0, C0]
                bounds = ([0, -10, 0.1, 0, -10, 0.1, 0], 
                         [np.inf, 10, 5, np.inf, 10, 5, np.inf])
                
            else:  # Gaussian
                fit_func = gaussian
                A0 = float(self.amp_entry.get())
                x00 = float(self.center_entry.get())
                sigma0 = float(self.width_entry.get())
                C0 = float(self.bg_entry.get())
                p0 = [A0, x00, sigma0, C0]
                bounds = ([0, -10, 0.1, 0], [np.inf, 10, 5, np.inf])
            
            # Perform fit
            popt, perr = fit_spectrum(x_data, y_data, y_errors, fit_func, p0, bounds)
            
            # Update fit results
            if len(self.fit_results) <= self.current_q_index:
                self.fit_results.extend([None] * (self.current_q_index - len(self.fit_results) + 1))
            self.fit_results[self.current_q_index] = (popt, perr)
            
            # Update plot with fit
            self.ax2.clear()
            mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_errors > 0)
            x_fit = x_data[mask]
            y_fit = y_data[mask]
            errors_fit = y_errors[mask]
            
            self.ax2.errorbar(x_fit, y_fit, yerr=errors_fit, 
                             fmt='o', markersize=4, alpha=0.7, label='Data',
                             capsize=2, elinewidth=1)
            
            x_fine = np.linspace(x_fit.min(), x_fit.max(), 500)
            self.ax2.plot(x_fine, fit_func(x_fine, *popt), 
                         'r-', linewidth=2, label=f'{model} Fit')
            
            self.ax2.set_title(f"Fit at Q = {q_val:.3f} Å⁻¹", fontsize=self.font_sizes['title'])
            self.ax2.set_xlabel("ω (meV)", fontsize=self.font_sizes['label'])
            self.ax2.set_ylabel("S(ω)", fontsize=self.font_sizes['label'])
            self.ax2.tick_params(axis='both', labelsize=self.font_sizes['tick'])
            self.ax2.legend(fontsize=self.font_sizes['legend'])
            self.ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Show parameters
            param_text = f"Model: {model}\n"
            param_names = ['Amplitude', 'Center', 'Width', 'Background']
            for i, (val, err) in enumerate(zip(popt, perr)):
                name = param_names[i] if i < len(param_names) else f'Param {i}'
                param_text += f"{name}: {val:.3f} ± {err:.3f}\n"
            
            self.ax2.text(0.02, 0.98, param_text, transform=self.ax2.transAxes,
                         verticalalignment='top', fontsize=self.font_sizes['info'],
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.canvas.draw()
            messagebox.showinfo("Fit Complete", "Current spectrum fitted successfully!")
            
        except Exception as e:
            messagebox.showerror("Fit Error", f"Fit failed: {str(e)}")
            traceback.print_exc()
    
    def fit_all(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        self.fit_results = []
        successful_fits = 0
        
        for i in range(len(self.data['q'])):
            try:
                # Get spectrum
                x_data = self.data['omega']
                y_data = self.data['S_data'][i, :]
                
                # Get errors if available
                if 'S_errors' in self.data and self.data['S_errors'].shape == self.data['S_data'].shape:
                    y_errors = self.data['S_errors'][i, :]
                else:
                    y_errors = np.ones_like(y_data) * 0.1
                
                # Use Lorentzian model for batch fitting
                A0 = np.max(y_data)
                C0 = np.min(y_data)
                
                # Try to estimate initial center from data
                mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_errors > 0)
                x_fit = x_data[mask]
                y_fit = y_data[mask]
                
                if len(x_fit) < 4:
                    self.fit_results.append(None)
                    continue
                
                center_idx = np.argmax(y_fit)
                x00 = x_fit[center_idx] if center_idx < len(x_fit) else 0
                gamma0 = 1.0
                
                p0 = [A0, x00, gamma0, C0]
                bounds = ([0, -10, 0.1, 0], [np.inf, 10, 5, np.inf])
                
                popt, pcov = fit_spectrum(x_data, y_data, y_errors, lorentzian, p0, bounds)
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
                self.fit_results.append((popt, perr))
                successful_fits += 1
                
            except Exception as e:
                print(f"Fit failed for Q index {i}: {e}")
                self.fit_results.append(None)
        
        # Update plots with fit results
        self.update_plots()
        
        messagebox.showinfo("Fit Complete", 
                          f"Fitted {successful_fits}/{len(self.data['q'])} spectra")
    
    def plot_fit_results(self):
        if len(self.fit_results) > 0:
            plot_fit_results(self.ax3, self.ax4, self.data, self.fit_results, self.font_sizes, self.canvas)
    
    def export_results(self):
        if self.data is None or len(self.fit_results) == 0:
            messagebox.showwarning("Warning", "No fit results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                export_results(filename, self.data, self.fit_results)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                traceback.print_exc()