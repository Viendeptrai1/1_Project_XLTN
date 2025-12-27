"""
Simplified Main GUI Application Window  
Audio Source Separation - FastICA + Single-Channel
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import numpy as np
import sounddevice as sd

from ..signal_processing import load_wav, save_wav, create_mixtures, pad_signals
from ..features import mfcc, lpc, stft
from ..ica import FastICA
from ..single_channel import SparseSeparation, SparseNMFSeparation
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
        """Create main GUI layout with 4 tabs"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_selection = ttk.Frame(notebook)
        self.tab_features = ttk.Frame(notebook)
        self.tab_separation = ttk.Frame(notebook)
        self.tab_single = ttk.Frame(notebook)
        
        notebook.add(self.tab_selection, text="1. Audio Selection & Mixing")
        notebook.add(self.tab_features, text="2. Feature Extraction")
        notebook.add(self.tab_separation, text="3. Multi-Channel Separation")
        notebook.add(self.tab_single, text="4. Single-Channel Separation")
        
        self._create_selection_tab()
        self._create_features_tab()
        self._create_separation_tab()
        self._create_single_channel_tab()
    
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
    
    def _create_features_tab(self):
        """Create feature extraction visualization tab"""
        # Parameters frame
        param_frame = ttk.LabelFrame(self.tab_features, text="Feature Extraction Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Audio file selector
        audio_frame = ttk.Frame(param_frame)
        audio_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(audio_frame, text="Select Audio File:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.feature_audio_var = tk.StringVar()
        if self.available_files:
            self.feature_audio_var.set(self.available_files[0])
        
        self.feature_audio_selector = ttk.Combobox(
            audio_frame,
            textvariable=self.feature_audio_var,
            values=self.available_files,
            state='readonly',
            width=40
        )
        self.feature_audio_selector.pack(side=tk.LEFT, padx=5)
        
        # Feature type selector
        type_frame = ttk.Frame(param_frame)
        type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(type_frame, text="Feature Type:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.feature_type_var = tk.StringVar(value='Both')
        
        ttk.Radiobutton(
            type_frame,
            text="MFCC",
            variable=self.feature_type_var,
            value='MFCC'
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            type_frame,
            text="LPC",
            variable=self.feature_type_var,
            value='LPC'
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            type_frame,
            text="STFT Spectrogram",
            variable=self.feature_type_var,
            value='STFT'
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            type_frame,
            text="All (Compare)",
            variable=self.feature_type_var,
            value='All'
        ).pack(side=tk.LEFT, padx=10)
        
        # Extract button
        button_frame = ttk.Frame(param_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame,
            text="🔬 Extract Features",
            command=self._extract_features
        ).pack(side=tk.LEFT, padx=5)
        
        self.feature_status_label = ttk.Label(button_frame, text="", foreground="blue")
        self.feature_status_label.pack(side=tk.LEFT, padx=10)
        
        # Plots frame
        plots_frame = ttk.LabelFrame(self.tab_features, text="Feature Visualization", padding=10)
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.plots_frame = plots_frame  # Store reference
        
        # Single feature canvas (full size) - for MFCC/LPC/STFT only
        self.plot_single_canvas = PlotCanvas(plots_frame, figsize=(14, 10))
        
        # Compare mode frame (3 rows)
        self.compare_frame = ttk.Frame(plots_frame)
        self.compare_frame.columnconfigure(0, weight=1)
        self.compare_frame.rowconfigure(0, weight=1)
        self.compare_frame.rowconfigure(1, weight=1)
        self.compare_frame.rowconfigure(2, weight=1)
        
        # Create three plot canvases for compare mode
        self.plot_mfcc_canvas = PlotCanvas(self.compare_frame, figsize=(14, 4))
        self.plot_mfcc_canvas.grid(row=0, column=0, sticky='nsew', pady=(0, 3))
        
        self.plot_lpc_canvas = PlotCanvas(self.compare_frame, figsize=(14, 4))
        self.plot_lpc_canvas.grid(row=1, column=0, sticky='nsew', pady=(3, 3))
        
        self.plot_stft_canvas = PlotCanvas(self.compare_frame, figsize=(14, 4))
        self.plot_stft_canvas.grid(row=2, column=0, sticky='nsew', pady=(3, 0))
        
        # Statistics display
        stats_frame = ttk.LabelFrame(self.tab_features, text="Feature Statistics & Comparison", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.feature_stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=120)
        self.feature_stats_text.pack(fill=tk.BOTH, expand=True)
        self.feature_stats_text.insert(tk.END, "Select an audio file and click 'Extract Features' to visualize features.")
        self.feature_stats_text.config(state=tk.DISABLED)
    
    def _extract_features(self):
        """Extract and visualize features from selected audio"""
        try:
            filename = self.feature_audio_var.get()
            if not filename:
                messagebox.showwarning("Warning", "Please select an audio file")
                return
            
            self.feature_status_label.config(text="Extracting features...", foreground="blue")
            self.root.update()
            
            # Load audio
            filepath = os.path.join(self.audio_dir, filename)
            signal, sr = load_wav(filepath)
            
            feature_type = self.feature_type_var.get()
            
            # Store for statistics
            self.current_mfcc = None
            self.current_lpc = None
            self.current_stft = None
            
            # Switch layout based on feature type
            if feature_type == 'All':
                # Compare mode: show 3 small canvases
                self.plot_single_canvas.pack_forget()
                self.compare_frame.pack(fill=tk.BOTH, expand=True)
                
                # Extract and plot all 3
                self.current_mfcc = mfcc(signal, sr, n_mfcc=13)
                self._plot_mfcc_heatmap(self.current_mfcc, filename)
                
                self.current_lpc = lpc(signal, sr, order=12)
                self._plot_lpc_heatmap(self.current_lpc, filename)
                
                self.current_stft = stft(signal)
                self._plot_stft_spectrogram(self.current_stft, filename, sr)
            else:
                # Single mode: show full-size canvas
                self.compare_frame.pack_forget()
                self.plot_single_canvas.pack(fill=tk.BOTH, expand=True)
                
                if feature_type == 'MFCC':
                    self.current_mfcc = mfcc(signal, sr, n_mfcc=13)
                    self._plot_single_feature('mfcc', self.current_mfcc, filename, sr)
                elif feature_type == 'LPC':
                    self.current_lpc = lpc(signal, sr, order=12)
                    self._plot_single_feature('lpc', self.current_lpc, filename, sr)
                elif feature_type == 'STFT':
                    self.current_stft = stft(signal)
                    self._plot_single_feature('stft', self.current_stft, filename, sr)
            
            # Update statistics
            self._display_feature_stats(signal, sr, filename)
            
            self.feature_status_label.config(text="✓ Extraction completed", foreground="green")
            
        except Exception as e:
            self.feature_status_label.config(text="❌ Error", foreground="red")
            messagebox.showerror("Error", f"Feature extraction failed:\\n{e}")
    
    def _plot_single_feature(self, feature_type, features, filename, sr):
        """Plot a single feature type in full-size canvas"""
        fig = self.plot_single_canvas.get_figure()
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        if feature_type == 'mfcc':
            n_mfcc, n_frames = features.shape
            im = ax.imshow(features, aspect='auto', origin='lower', 
                          cmap='viridis', interpolation='nearest')
            
            # X-axis: time in seconds
            frame_rate = 100
            time_ticks = ax.get_xticks()
            time_labels = [f'{t/frame_rate:.1f}s' for t in time_ticks if 0 <= t < n_frames]
            ax.set_xticks(time_ticks[:len(time_labels)])
            ax.set_xticklabels(time_labels, fontsize=11)
            
            # Y-axis: MFCC coefficient numbers
            ax.set_yticks(range(0, n_mfcc, 1))
            ax.set_yticklabels([f'C{i}' for i in range(0, n_mfcc, 1)], fontsize=11)
            
            ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_ylabel('MFCC Coefficient', fontsize=14, fontweight='bold')
            ax.set_title(f'MFCC Heatmap: {filename}\n({n_mfcc} coefficients × {n_frames} frames)', 
                        fontsize=16, fontweight='bold', pad=15)
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('MFCC Value', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
        elif feature_type == 'lpc':
            n_frames, order = features.shape
            im = ax.imshow(features.T, aspect='auto', origin='lower',
                          cmap='plasma', interpolation='nearest')
            
            # X-axis: time in seconds
            hop_length = 160
            frame_rate = sr / hop_length
            time_ticks = ax.get_xticks()
            time_labels = [f'{t/frame_rate:.2f}s' for t in time_ticks if 0 <= t < n_frames]
            ax.set_xticks(time_ticks[:len(time_labels)])
            ax.set_xticklabels(time_labels, fontsize=11)
            
            # Y-axis: LPC coefficient numbers
            ax.set_yticks(range(order))
            ax.set_yticklabels([f'a{i+1}' for i in range(order)], fontsize=11)
            
            ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_ylabel('LPC Coefficient', fontsize=14, fontweight='bold')
            ax.set_title(f'LPC Heatmap: {filename}\n({order} coefficients × {n_frames} frames)', 
                        fontsize=16, fontweight='bold', pad=15)
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Coefficient Value', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
        elif feature_type == 'stft':
            freq_bins, time_frames = features.shape
            magnitude = np.abs(features)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            
            im = ax.imshow(magnitude_db, aspect='auto', origin='lower',
                          cmap='inferno', interpolation='nearest', vmin=-60, vmax=0)
            
            # X-axis: time in seconds
            hop_length = 256
            frame_rate = sr / hop_length
            time_ticks = ax.get_xticks()
            time_labels = [f'{t/frame_rate:.2f}s' for t in time_ticks if 0 <= t < time_frames]
            ax.set_xticks(time_ticks[:len(time_labels)])
            ax.set_xticklabels(time_labels, fontsize=11)
            
            # Y-axis: frequency in Hz
            n_fft = 512
            freq_resolution = sr / n_fft
            freq_ticks = ax.get_yticks()
            freq_labels = [f'{int(t * freq_resolution)}Hz' for t in freq_ticks if 0 <= t < freq_bins]
            ax.set_yticks(freq_ticks[:len(freq_labels)])
            ax.set_yticklabels(freq_labels, fontsize=11)
            
            ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
            ax.set_title(f'STFT Spectrogram: {filename}\n({freq_bins} frequency bins × {time_frames} time frames)', 
                        fontsize=16, fontweight='bold', pad=15)
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Power (dB)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='white')
        
        fig.tight_layout()
        self.plot_single_canvas.update_plot()
    
    def _plot_mfcc_heatmap(self, mfcc_features, filename):
        """Plot MFCC features as heatmap"""
        fig = self.plot_mfcc_canvas.get_figure()
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        n_mfcc, n_frames = mfcc_features.shape
        
        # MFCC shape is (n_mfcc, n_frames), perfect for imshow
        im = ax.imshow(mfcc_features, aspect='auto', origin='lower', 
                      cmap='viridis', interpolation='nearest')
        
        # Better X-axis: show time in seconds
        frame_rate = 100  # Typical: 100 frames/second
        time_ticks = ax.get_xticks()
        time_labels = [f'{t/frame_rate:.1f}s' for t in time_ticks if 0 <= t < n_frames]
        ax.set_xticks(time_ticks[:len(time_labels)])
        ax.set_xticklabels(time_labels)
        
        # Better Y-axis: show MFCC coefficient numbers
        ax.set_yticks(range(0, n_mfcc, 2))  # Show every 2nd coefficient
        ax.set_yticklabels([f'C{i}' for i in range(0, n_mfcc, 2)])
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MFCC Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f'MFCC Heatmap: {filename}\n({n_mfcc} coeffs × {n_frames} frames)', 
                    fontsize=13, fontweight='bold', pad=12)
        
        # Add colorbar with better label
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('MFCC Value', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        # Grid for easier reading
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='white')
        
        fig.tight_layout()
        self.plot_mfcc_canvas.update_plot()
    
    def _plot_lpc_heatmap(self, lpc_features, filename):
        """Plot LPC features as heatmap"""
        fig = self.plot_lpc_canvas.get_figure()
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        n_frames, order = lpc_features.shape
        
        # LPC shape is (n_frames, order), need transpose for proper display
        im = ax.imshow(lpc_features.T, aspect='auto', origin='lower',
                      cmap='plasma', interpolation='nearest')
        
        # Better X-axis: show time in seconds  
        hop_length = 160
        sr = 16000
        frame_rate = sr / hop_length  # ~100 frames/second
        time_ticks = ax.get_xticks()
        time_labels = [f'{t/frame_rate:.2f}s' for t in time_ticks if 0 <= t < n_frames]
        ax.set_xticks(time_ticks[:len(time_labels)])
        ax.set_xticklabels(time_labels)
        
        # Better Y-axis: show LPC coefficient numbers
        ax.set_yticks(range(order))
        ax.set_yticklabels([f'a{i+1}' for i in range(order)])
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('LPC Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f'LPC Heatmap: {filename}\n({order} coeffs × {n_frames} frames)', 
                    fontsize=13, fontweight='bold', pad=12)
        
        # Add colorbar with better label
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Coefficient Value', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='white')
        
        fig.tight_layout()
        self.plot_lpc_canvas.update_plot()
    
    def _plot_stft_spectrogram(self, stft_matrix, filename, sr):
        """Plot STFT spectrogram"""
        fig = self.plot_stft_canvas.get_figure()
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        freq_bins, time_frames = stft_matrix.shape
        
        # STFT returns complex values, take magnitude and convert to dB
        magnitude = np.abs(stft_matrix)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
        
        # Plot spectrogram
        im = ax.imshow(magnitude_db, aspect='auto', origin='lower',
                      cmap='inferno', interpolation='nearest', vmin=-60, vmax=0)
        
        # Better X-axis: show time in seconds
        hop_length = 256  # Default STFT hop
        frame_rate = sr / hop_length
        time_ticks = ax.get_xticks()
        time_labels = [f'{t/frame_rate:.2f}s' for t in time_ticks if 0 <= t < time_frames]
        ax.set_xticks(time_ticks[:len(time_labels)])
        ax.set_xticklabels(time_labels)
        
        # Better Y-axis: show frequency in Hz
        n_fft = 512  # Default
        freq_resolution = sr / n_fft
        freq_ticks = ax.get_yticks()
        freq_labels = [f'{int(t * freq_resolution)}Hz' for t in freq_ticks if 0 <= t < freq_bins]
        ax.set_yticks(freq_ticks[:len(freq_labels)])
        ax.set_yticklabels(freq_labels)
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_title(f'STFT Spectrogram: {filename}\n({freq_bins} bins × {time_frames} frames)', 
                    fontsize=13, fontweight='bold', pad=12)
        
        # Add colorbar with better range
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Power (dB)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='white')
        
        fig.tight_layout()
        self.plot_stft_canvas.update_plot()
    
    def _display_feature_stats(self, signal, sr, filename):
        """Display feature statistics and comparison"""
        self.feature_stats_text.config(state=tk.NORMAL)
        self.feature_stats_text.delete(1.0, tk.END)
        
        # Audio info
        duration = len(signal) / sr
        self.feature_stats_text.insert(tk.END, f"=== Audio File: {filename} ===\\n")
        self.feature_stats_text.insert(tk.END, f"Duration: {duration:.2f}s  |  Sample Rate: {sr}Hz  |  Samples: {len(signal):,}\\n\\n")
        
        # MFCC stats
        if self.current_mfcc is not None:
            n_mfcc, n_frames = self.current_mfcc.shape
            self.feature_stats_text.insert(tk.END, "--- MFCC Features ---\\n")
            self.feature_stats_text.insert(tk.END, f"Shape: ({n_mfcc} coefficients, {n_frames} frames)\\n")
            self.feature_stats_text.insert(tk.END, f"Mean: {np.mean(self.current_mfcc):.4f}  |  Std: {np.std(self.current_mfcc):.4f}\\n")
            self.feature_stats_text.insert(tk.END, f"Min: {np.min(self.current_mfcc):.4f}  |  Max: {np.max(self.current_mfcc):.4f}\\n\\n")
        
        # LPC stats
        if self.current_lpc is not None:
            n_frames, order = self.current_lpc.shape
            self.feature_stats_text.insert(tk.END, "--- LPC Features ---\\n")
            self.feature_stats_text.insert(tk.END, f"Shape: ({n_frames} frames, {order} coefficients)\\n")
            self.feature_stats_text.insert(tk.END, f"Mean: {np.mean(self.current_lpc):.4f}  |  Std: {np.std(self.current_lpc):.4f}\\n")
            self.feature_stats_text.insert(tk.END, f"Min: {np.min(self.current_lpc):.4f}  |  Max: {np.max(self.current_lpc):.4f}\\n\\n")
        
        # STFT stats
        if self.current_stft is not None:
            freq_bins, time_frames = self.current_stft.shape
            magnitude = np.abs(self.current_stft)
            self.feature_stats_text.insert(tk.END, "--- STFT Spectrogram ---\\n")
            self.feature_stats_text.insert(tk.END, f"Shape: ({freq_bins} frequency bins, {time_frames} time frames)\\n")
            self.feature_stats_text.insert(tk.END, f"Magnitude Mean: {np.mean(magnitude):.4f}  |  Std: {np.std(magnitude):.4f}\\n")
            self.feature_stats_text.insert(tk.END, f"Magnitude Min: {np.min(magnitude):.4f}  |  Max: {np.max(magnitude):.4f}\\n\\n")
        
        # Comparison
        if sum([self.current_mfcc is not None, self.current_lpc is not None, self.current_stft is not None]) >= 2:
            self.feature_stats_text.insert(tk.END, "--- Comparison ---\\n")
            if self.current_mfcc is not None:
                self.feature_stats_text.insert(tk.END, f"✓ MFCC: {self.current_mfcc.shape[0]} coefficients (mel-frequency cepstral, perceptual)\\n")
            if self.current_lpc is not None:
                self.feature_stats_text.insert(tk.END, f"✓ LPC: {self.current_lpc.shape[1]} coefficients (vocal tract model, speech-specific)\\n")
            if self.current_stft is not None:
                self.feature_stats_text.insert(tk.END, f"✓ STFT: {self.current_stft.shape[0]} frequency bins (time-frequency representation)\\n")
            self.feature_stats_text.insert(tk.END, f"✓ All methods capture temporal evolution through frames\\n")
        
        self.feature_stats_text.config(state=tk.DISABLED)
    
    def _create_separation_tab(self):
        """Create FastICA separation tab"""
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
            text="▶️ Run Separation",
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
            
            # Plot - Enhanced intelligent visualization
            fig = self.plot_mixing.get_figure()
            fig.clear()
            
            n_sources = len(sources)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Show up to 5 sources for readability
            max_display = min(5, n_sources)
            
            for i in range(max_display):
                ax = fig.add_subplot(max_display, 1, i+1)
                
                # Get signal info
                signal = sources[i]
                n_samples = len(signal)
                duration = n_samples / self.sample_rate
                
                # Intelligent downsampling for smooth display (keep max 10000 points)
                if n_samples > 10000:
                    step = max(1, n_samples // 10000)
                    display_signal = signal[::step]
                    time_axis = np.arange(0, n_samples, step) / self.sample_rate
                else:
                    display_signal = signal
                    time_axis = np.arange(n_samples) / self.sample_rate
                
                # Plot waveform with color
                ax.plot(time_axis, display_signal, color=colors[i % len(colors)], 
                       linewidth=0.7, alpha=0.85)
                
                # Add subtle envelope for better visualization
                if len(display_signal) > 100:
                    envelope = np.abs(display_signal)
                    ax.fill_between(time_axis, -envelope, envelope, 
                                   alpha=0.15, color=colors[i % len(colors)])
                
                # Styling
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
                ax.set_ylabel("Amplitude", fontsize=9, fontweight='bold')
                
                # Set y-limits with 10% padding
                y_max = np.abs(signal).max()
                ax.set_ylim([-1.15 * y_max, 1.15 * y_max])
                
                # Title with metadata (no emojis for font compatibility)
                title_text = f"{selected_files[i]}  |  {duration:.2f}s  |  {self.sample_rate}Hz  |  {n_samples:,} samples"
                ax.set_title(title_text, fontsize=9, fontweight='bold', pad=8)
                
                # Only show x-label on bottom subplot
                if i == max_display - 1:
                    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
                else:
                    ax.set_xticklabels([])
            
            # Overall title
            if n_sources > max_display:
                fig.suptitle(f"Waveform Preview — Showing {max_display} of {n_sources} Sources", 
                           fontsize=12, fontweight='bold', y=0.998)
            else:
                fig.suptitle(f"Waveform Preview — {n_sources} Source{'s' if n_sources > 1 else ''}", 
                           fontsize=12, fontweight='bold', y=0.998)
            
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            self.plot_mixing.update_plot()
            
            # Update Tab 3 mixture selector
            self._update_mixture_selector()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate mixtures:\\n{e}")
    
    def _run_fastica(self):
        """Run FastICA separation"""
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
            messagebox.showerror("Error", f"FastICA failed:\n{e}")
    
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
    def _create_single_channel_tab(self):
        """Create single-channel separation tab"""
        # Parameters (no instructions)
        param_frame = ttk.LabelFrame(self.tab_single, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Mixture selector
        mix_frame = ttk.Frame(param_frame)
        mix_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mix_frame, text="Select Mixture:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.mixture_selector_var = tk.StringVar(value="Mixture 1")
        self.mixture_selector = ttk.Combobox(
            mix_frame,
            textvariable=self.mixture_selector_var,
            state='readonly',
            width=30
        )
        self.mixture_selector.pack(side=tk.LEFT, padx=5)
        
        # Number of sources
        sources_frame = ttk.Frame(param_frame)
        sources_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sources_frame, text="Number of Sources:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.n_sources_var = tk.IntVar(value=2)
        ttk.Spinbox(
            sources_frame,
            from_=2,
            to=5,
            textvariable=self.n_sources_var,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Method selector
        method_frame = ttk.Frame(param_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="Method:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.single_method_var = tk.StringVar(value='Binary Masking')
        
        ttk.Radiobutton(
            method_frame,
            text="Binary Masking (Fast, K-means)",
            variable=self.single_method_var,
            value='Binary Masking'
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            method_frame,
            text="Sparse NMF (Better quality)",
            variable=self.single_method_var,
            value='Sparse NMF'
        ).pack(side=tk.LEFT, padx=10)
        
        # Run button
        run_frame = ttk.Frame(param_frame)
        run_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            run_frame,
            text="▶️ Run Single-Channel Separation",
            command=self._run_single_channel
        ).pack(side=tk.LEFT, padx=5)
        
        self.single_status_label = ttk.Label(run_frame, text="", foreground="blue")
        self.single_status_label.pack(side=tk.LEFT, padx=10)
        
        # Results
        results_frame = ttk.LabelFrame(self.tab_single, text="Separated Sources", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.single_results_text = scrolledtext.ScrolledText(results_frame, height=20, width=100)
        self.single_results_text.pack(fill=tk.BOTH, expand=True)
    def _update_mixture_selector(self):
        """Update mixture selector dropdown after mixtures are generated"""
        if self.mixtures is not None:
            n_mixtures = len(self.mixtures)
            values = [f"Mixture {i+1}" for i in range(n_mixtures)]
            self.mixture_selector['values'] = values
            if values:
                self.mixture_selector.current(0)
    
    def _run_single_channel(self):
        """Run single-channel separation"""
        if self.mixtures is None:
            messagebox.showwarning("Warning", "Please generate mixtures first (Tab 1)")
            return
        
        # Get selected mixture
        mix_idx = int(self.mixture_selector_var.get().split()[-1]) - 1
        mixture = self.mixtures[mix_idx]
        
        n_sources = self.n_sources_var.get()
        method = self.single_method_var.get()
        
        self.single_status_label.config(text=f"Running {method}...", foreground="blue")
        self.root.update()
        
        try:
            # Run separation
            if method == 'Binary Masking':
                separator = SparseSeparation(n_sources=n_sources)
            else:  # Sparse NMF
                separator = SparseNMFSeparation(n_sources=n_sources)
            
            sources = separator.separate(mixture, sr=self.sample_rate)
            
            # Store results
            self.single_sources = sources
            
            self.single_status_label.config(text=f"✓ {method} completed", foreground="green")
            
            # Display results
            self._display_single_results()
            
            messagebox.showinfo("Success", f"Separated 1 mixture into {len(sources)} sources!")
            
        except Exception as e:
            self.single_status_label.config(text="❌ Error", foreground="red")
            messagebox.showerror("Error", f"{method} failed:\\n{e}")
    
    def _display_single_results(self):
        """Display single-channel separation results"""
        self.single_results_text.config(state=tk.NORMAL)
        self.single_results_text.delete(1.0, tk.END)
        
        self.single_results_text.insert(tk.END, "=== Single-Channel Separation Results ===\n\n")
        
        mix_idx = int(self.mixture_selector_var.get().split()[-1]) - 1
        self.single_results_text.insert(tk.END, f"Input: Mixture {mix_idx+1}\n")
        self.single_results_text.insert(tk.END, f"Method: {self.single_method_var.get()}\n")
        self.single_results_text.insert(tk.END, f"Number of sources: {len(self.single_sources)}\n\n")
        
        # Show each source with play button
        for i, source in enumerate(self.single_sources):
            energy = np.sum(source ** 2)
            
            self.single_results_text.insert(tk.END, f"Source {i+1}:\n")
            self.single_results_text.insert(tk.END, f"  Energy: {energy:.2e}\n")
            self.single_results_text.insert(tk.END, f"  Duration: {len(source) / self.sample_rate:.2f}s\n")
            
            # Play button
            btn = ttk.Button(
                self.single_results_text,
                text=f"▶️ Play Source {i+1}",
                command=lambda idx=i: self._play_single_source(idx)
            )
            self.single_results_text.window_create(tk.END, window=btn)
            self.single_results_text.insert(tk.END, "\n\n")
        
        self.single_results_text.insert(tk.END, "=" * 60 + "\n")
        self.single_results_text.insert(tk.END, "Note: SNR will be lower than multi-channel (3-8 dB)\n")
        self.single_results_text.insert(tk.END, "This is expected for single-channel separation!\n")
        
        self.single_results_text.config(state=tk.DISABLED)
    
    def _play_single_source(self, source_idx):
        """Play separated source from single-channel"""
        if not hasattr(self, 'single_sources'):
            return
        
        try:
            sd.stop()
            sd.play(self.single_sources[source_idx], self.sample_rate)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play source:\\n{e}")
