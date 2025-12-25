"""
Simplified Main GUI Application Window
Audio Source Separation System using FastICA
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import numpy as np
import sounddevice as sd

from ..signal_processing import load_wav, save_wav, create_mixtures, pad_signals
from ..features import mfcc
from ..ica import FastICA
from ..evaluation import snr, sdr, permutation_solver
from ..recognition import DTWClassifier
from ..visualization import plot_comparison
from .plot_canvas import PlotCanvas


class AudioSeparationApp:
    """
    Simplified application for audio source separation
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Source Separation - FastICA")
        self.root.geometry("1400x900")
        
        # Audio dataset directory
        self.audio_dir = "tts_dataset_vi/"
        
        # State variables
        self.available_files = []
        self.selected_files = []
        self.file_vars = []  # Checkbox variables
        self.audio_data_cache = {}  # Cache loaded audio
        
        self.mixtures = None
        self.mixing_matrix = None
        self.separated_sources = None
        self.ica_model = None
        self.sample_rate = 16000
        
        # Load available audio files
        self._load_available_files()
        
        # Create GUI
        self._create_widgets()
    
    def _load_available_files(self):
        """Load list of available audio files"""
        if not os.path.exists(self.audio_dir):
            messagebox.showerror("Error", f"Audio directory not found: {self.audio_dir}")
            return
        
        # Load ALL audio files
        self.available_files = sorted([f for f in os.listdir(self.audio_dir) 
                                      if f.endswith('.wav')])
    
    def _create_widgets(self):
        """Create main GUI layout with 2 tabs"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_selection = ttk.Frame(notebook)
        self.tab_separation = ttk.Frame(notebook)
        
        notebook.add(self.tab_selection, text="1. Audio Selection & Mixing")
        notebook.add(self.tab_separation, text="2. ICA Separation & Results")
        
        self._create_selection_tab()
        self._create_separation_tab()
    
    def _create_selection_tab(self):
        """Create audio selection tab with checkboxes"""
        # Main content frame - COMPACT!
        main_frame = ttk.Frame(self.tab_selection, height=250)  # Fixed small height
        main_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        main_frame.pack_propagate(False)  # Don't expand
        
        # Left: File selection
        left_frame = ttk.LabelFrame(main_frame, text="Available Audio Files (Select 2-5)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Scrollable file list
        canvas = tk.Canvas(left_frame, height=400)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create checkboxes for each file
        self.file_vars = []
        for filename in self.available_files:
            var = tk.IntVar()
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            cb = ttk.Checkbutton(
                frame, 
                text=filename,
                variable=var,
                command=self._on_selection_changed
            )
            cb.pack(side=tk.LEFT, padx=5)
            
            btn_play = ttk.Button(
                frame,
                text="▶️",
                width=3,
                command=lambda f=filename: self._play_file(f)
            )
            btn_play.pack(side=tk.LEFT, padx=2)
            
            self.file_vars.append((filename, var))
        
        # Selection info
        self.selection_label = ttk.Label(left_frame, text="0 files selected", foreground="gray")
        self.selection_label.pack(pady=5)
        
        # Right: Mixtures preview
        right_frame = ttk.LabelFrame(main_frame, text="Generated Mixtures", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.mixtures_text = scrolledtext.ScrolledText(right_frame, height=8, width=25)  # Smaller!
        self.mixtures_text.pack(fill=tk.BOTH, expand=True)
        self.mixtures_text.insert(tk.END, "No mixtures generated yet.\nSelect 2-5 files to start.")
        self.mixtures_text.config(state=tk.DISABLED)
        
        # Mixture play buttons frame
        self.mixture_buttons_frame = ttk.Frame(right_frame)
        self.mixture_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Plot frame - BIGGER!
        plot_frame = ttk.LabelFrame(self.tab_selection, text="Waveform Preview", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.plot_mixing = PlotCanvas(plot_frame, figsize=(20, 12))  # HUGE!
        self.plot_mixing.pack(fill=tk.BOTH, expand=True)
    
    def _create_separation_tab(self):
        """Create ICA separation tab with auto-results"""
        # Parameters
        param_frame = ttk.LabelFrame(self.tab_separation, text="FastICA Parameters", padding=10)
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Max Iterations:").pack(side=tk.LEFT, padx=5)
        self.max_iter_var = tk.IntVar(value=200)
        ttk.Entry(param_frame, textvariable=self.max_iter_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text="Tolerance:").pack(side=tk.LEFT, padx=5)
        self.tol_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(param_frame, textvariable=self.tol_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            param_frame,
            text="▶️ Run FastICA",
            command=self._run_fastica
        ).pack(side=tk.LEFT, padx=20)
        
        self.status_label = ttk.Label(param_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Top row: Separated Sources (left) + Recognition (right)
        top_results_frame = ttk.Frame(self.tab_separation)
        top_results_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Left: Separated sources with play buttons
        left_results = ttk.LabelFrame(top_results_frame, text="Separated Sources", padding=10)
        left_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.sources_text = scrolledtext.ScrolledText(left_results, height=12, width=40)
        self.sources_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Recognition results
        right_results = ttk.LabelFrame(top_results_frame, text="Recognition Results (DTW)", padding=10)
        right_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.recognition_text = scrolledtext.ScrolledText(right_results, height=12, width=40)
        self.recognition_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Evaluation Metrics (full width)
        metrics_frame = ttk.LabelFrame(self.tab_separation, text="Evaluation Metrics (SNR & SDR)", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=8, width=100)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def _on_selection_changed(self):
        """Handle checkbox selection changes"""
        selected = [filename for filename, var in self.file_vars if var.get() == 1]
        
        # Update label
        count = len(selected)
        if count < 2:
            self.selection_label.config(text=f"{count} files selected (need 2-5)", foreground="red")
        elif count > 5:
            self.selection_label.config(text=f"{count} files selected (max 5)", foreground="red")
        else:
            self.selection_label.config(text=f"✓ {count} files selected", foreground="green")
        
        # Disable checkboxes if 5 selected
        for filename, var in self.file_vars:
            if var.get() == 0 and count >= 5:
                # Find and disable checkbox
                pass  # Will handle in next iteration
        
        # Auto-generate mixtures if valid selection
        if 2 <= count <= 5:
            self._generate_mixtures(selected)
    
    def _play_file(self, filename):
        """Play audio file"""
        try:
            filepath = os.path.join(self.audio_dir, filename)
            
            # Load from cache or file
            if filename not in self.audio_data_cache:
                data, sr = load_wav(filepath)
                self.audio_data_cache[filename] = (data, sr)
            else:
                data, sr = self.audio_data_cache[filename]
            
            # Play
            sd.stop()  # Stop any playing audio
            sd.play(data, sr)
            
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play {filename}:\\n{e}")
    
    def _play_mixture(self, mix_idx):
        """Play mixture"""
        if self.mixtures is None:
            return
        
        try:
            sd.stop()
            sd.play(self.mixtures[mix_idx], self.sample_rate)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play mixture:\\n{e}")
    
    def _play_separated(self, source_idx):
        """Play separated source"""
        if self.separated_sources is None:
            return
        
        try:
            sd.stop()
            sd.play(self.separated_sources[source_idx], self.sample_rate)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play source:\\n{e}")
    
    def _generate_mixtures(self, selected_files):
        """Automatically generate mixtures"""
        try:
            # Load selected files
            sources = []
            for filename in selected_files:
                if filename not in self.audio_data_cache:
                    filepath = os.path.join(self.audio_dir, filename)
                    data, sr = load_wav(filepath)
                    self.audio_data_cache[filename] = (data, sr)
                    self.sample_rate = sr
                else:
                    data, sr = self.audio_data_cache[filename]
                
                sources.append(data)
            
            self.selected_files = selected_files
            self.audio_data = sources
            
            # Create mixtures
            self.mixtures, self.mixing_matrix = create_mixtures(sources)
            
            # Update mixtures display
            self.mixtures_text.config(state=tk.NORMAL)
            self.mixtures_text.delete(1.0, tk.END)
            self.mixtures_text.insert(tk.END, f"✓ Generated {len(self.mixtures)} mixtures\n\n")
            self.mixtures_text.insert(tk.END, "Mixing Matrix:\n")
            self.mixtures_text.insert(tk.END, np.array2string(self.mixing_matrix, precision=2, suppress_small=True))
            self.mixtures_text.config(state=tk.DISABLED)
            
            # Create play buttons for mixtures
            for widget in self.mixture_buttons_frame.winfo_children():
                widget.destroy()
            
            for i in range(len(self.mixtures)):
                btn = ttk.Button(
                    self.mixture_buttons_frame,
                    text=f"▶️ Mixture {i+1}",
                    command=lambda idx=i: self._play_mixture(idx)
                )
                btn.pack(side=tk.LEFT, padx=5)
            
            # Plot
            fig = self.plot_mixing.get_figure()
            fig.clear()
            
            # Simple waveform plot
            n_sources = len(sources)
            for i in range(min(3, n_sources)):  # Show first 3
                ax = fig.add_subplot(3, 1, i+1)
                ax.plot(sources[i][:5000])  # First 5000 samples
                ax.set_title(f"Source: {selected_files[i]}")
                ax.set_ylabel("Amplitude")
            
            fig.tight_layout()
            self.plot_mixing.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate mixtures:\\n{e}")
    
    def _run_fastica(self):
        """Run FastICA and auto-compute results"""
        if self.mixtures is None:
            messagebox.showwarning("Warning", "Please select audio files first (Tab 1)")
            return
        
        self.status_label.config(text="Running FastICA...", foreground="blue")
        self.root.update()
        
        try:
            # Run FastICA
            self.ica_model = FastICA(
                n_components=len(self.mixtures),
                max_iter=self.max_iter_var.get(),
                tol=self.tol_var.get(),
                random_state=42
            )
            
            self.separated_sources = self.ica_model.fit_transform(self.mixtures)
            
            # Align with originals
            sources_padded = pad_signals(self.audio_data)
            aligned_sources, permutation, correlations = permutation_solver(
                sources_padded,
                self.separated_sources
            )
            self.separated_sources = aligned_sources
            
            self.status_label.config(
                text=f"✓ FastICA completed in {self.ica_model.n_iter} iterations",
                foreground="green"
            )
            
            
            # Auto-compute recognition and metrics
            self._auto_recognition()
            self._auto_metrics()
            
            messagebox.showinfo("Success", "Separation completed! Check results above.")
            
        except Exception as e:
            self.status_label.config(text="❌ Error", foreground="red")
            messagebox.showerror("Error", f"FastICA failed:\\n{e}")
    
    def _auto_recognition(self):
        """Automatically recognize separated sources"""
        # Separated sources panel
        self.sources_text.config(state=tk.NORMAL)
        self.sources_text.delete(1.0, tk.END)
        self.sources_text.insert(tk.END, "=== Separated Sources ===\n\n")
        
        # Recognition results panel
        self.recognition_text.config(state=tk.NORMAL)
        self.recognition_text.delete(1.0, tk.END)
        self.recognition_text.insert(tk.END, "=== Recognition Results ===\n\n")
        
        # Create DTW classifier with selected files as templates
        template_mfcc = []
        template_labels = []
        
        for i, filename in enumerate(self.selected_files):
            data = self.audio_data[i]
            mfcc_feat = mfcc(data, self.sample_rate)
            template_mfcc.append(mfcc_feat.T)
            # Extract label from filename (e.g., "digit_0" -> "0")
            label = filename.replace('.wav', '').split('_')[1]
            template_labels.append(label)
        
        clf = DTWClassifier()
        clf.fit(template_mfcc, template_labels)
        
        # Recognize each separated source
        for i, source in enumerate(self.separated_sources):
            mfcc_feat = mfcc(source, self.sample_rate)
            predicted, distance = clf.predict_single(mfcc_feat.T)
            
            # Expected label
            expected = template_labels[i]
            status = "✓" if predicted == expected else "✗"
            
            # Add to sources panel (with play button)
            self.sources_text.insert(tk.END, f"Source {i+1}:\n")
            self.sources_text.insert(tk.END, f"  File: {self.selected_files[i]}\n")
            
            btn = ttk.Button(
                self.sources_text,
                text=f"▶️ Play Source {i+1}",
                command=lambda idx=i: self._play_separated(idx)
            )
            self.sources_text.window_create(tk.END, window=btn)
            self.sources_text.insert(tk.END, "\n\n")
            
            # Add to recognition panel
            self.recognition_text.insert(tk.END, f"Source {i+1}:\n")
            self.recognition_text.insert(tk.END, f"  Expected: {expected}\n")
            self.recognition_text.insert(tk.END, f"  {status} Predicted: {predicted}\n")
            self.recognition_text.insert(tk.END, f"  DTW Distance: {distance:.2f}\n\n")
        
        self.sources_text.config(state=tk.DISABLED)
        self.recognition_text.config(state=tk.DISABLED)
    
    def _auto_metrics(self):
        """Automatically compute evaluation metrics"""
        sources_padded = pad_signals(self.audio_data)
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "=== Evaluation Metrics ===\n\n")
        
        snr_values = []
        sdr_values = []
        
        for i in range(len(sources_padded)):
            snr_val = snr(sources_padded[i], self.separated_sources[i])
            sdr_val = sdr(sources_padded[i], self.separated_sources[i])
            
            snr_values.append(snr_val)
            sdr_values.append(sdr_val)
            
            status = "✅" if snr_val > 10 else ("⚠️" if snr_val > 5 else "❌")
            
            self.metrics_text.insert(tk.END, f"Source {i+1}: {self.selected_files[i]}\n")
            self.metrics_text.insert(tk.END, f"  SNR: {snr_val:.2f} dB {status}\n")
            self.metrics_text.insert(tk.END, f"  SDR: {sdr_val:.2f} dB\n\n")
        
        avg_snr = np.mean(snr_values)
        avg_sdr = np.mean(sdr_values)
        
        overall_status = "✅ Excellent" if avg_snr > 10 else ("⚠️ Fair" if avg_snr > 5 else "❌ Poor")
        
        self.metrics_text.insert(tk.END, "=" * 40 + "\n")
        self.metrics_text.insert(tk.END, f"Average SNR: {avg_snr:.2f} dB\n")
        self.metrics_text.insert(tk.END, f"Average SDR: {avg_sdr:.2f} dB\n")
        self.metrics_text.insert(tk.END, f"Overall Quality: {overall_status}\n")
        
        self.metrics_text.config(state=tk.DISABLED)
    
    def _plot_results(self):
        """Plot separation results"""
        fig = self.plot_separation.get_figure()
        fig.clear()
        
        try:
            plot_comparison(
                self.audio_data,
                self.mixtures,
                self.separated_sources,
                self.sample_rate,
                titles=[f.replace('.wav', '') for f in self.selected_files]
            )
            self.plot_separation.update_plot()
        except Exception as e:
            print(f"Plot error: {e}")
            # Simple fallback plot
            n = len(self.separated_sources)
            for i in range(min(3, n)):
                ax = fig.add_subplot(3, 1, i+1)
                ax.plot(self.separated_sources[i][:5000])
                ax.set_title(f"Separated: {self.selected_files[i]}")
                ax.set_ylabel("Amplitude")
            fig.tight_layout()
            self.plot_separation.update_plot()
