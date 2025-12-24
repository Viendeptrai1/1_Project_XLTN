"""
Main GUI Application Window
Audio Source Separation System using FastICA and NMF
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np

from ..signal_processing import load_wav, save_wav, create_mixtures, pad_signals
from ..features import mfcc
from ..ica import FastICA
from ..nmf import NMF
from ..evaluation import snr, sdr, permutation_solver
from ..recognition import DTWClassifier
from ..visualization import plot_waveform, plot_spectrogram, plot_mixing_matrix, plot_comparison
from .audio_player import AudioPlayer
from .plot_canvas import PlotCanvas


class AudioSeparationApp:
    """
    Main application for audio source separation
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Source Separation - FastICA & NMF")
        self.root.geometry("1200x800")
        
        self.audio_files = []
        self.audio_data = []
        self.sample_rates = []
        self.mixtures = None
        self.mixing_matrix = None
        self.separated_sources = None
        self.ica_model = None
        
        # NMF variables
        self.nmf_model = None
        self.nmf_separated = None
        
        self.dtw_classifier = None
        self.templates_mfcc = []
        self.templates_labels = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create main GUI layout"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_mixing = ttk.Frame(notebook)
        self.tab_separation = ttk.Frame(notebook)
        self.tab_nmf = ttk.Frame(notebook)
        self.tab_recognition = ttk.Frame(notebook)
        self.tab_evaluation = ttk.Frame(notebook)
        
        notebook.add(self.tab_mixing, text="1. Mixing")
        notebook.add(self.tab_separation, text="2. ICA Separation")
        notebook.add(self.tab_nmf, text="3. NMF Separation")
        notebook.add(self.tab_recognition, text="4. Recognition")
        notebook.add(self.tab_evaluation, text="5. Evaluation")
        
        self._create_mixing_tab()
        self._create_separation_tab()
        self._create_nmf_tab()
        self._create_recognition_tab()
        self._create_evaluation_tab()
    
    def _create_mixing_tab(self):
        """Create mixing tab"""
        control_frame = ttk.LabelFrame(self.tab_mixing, text="Audio Selection", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Select Audio Files (4-5 files)", 
                  command=self.select_audio_files).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Mixtures", 
                  command=self.generate_mixtures).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Mixtures", 
                  command=self.save_mixtures).pack(side=tk.LEFT, padx=5)
        
        self.file_list_label = ttk.Label(control_frame, text="No files selected")
        self.file_list_label.pack(side=tk.LEFT, padx=10)
        
        self.plot_mixing = PlotCanvas(self.tab_mixing, figsize=(12, 6))
        self.plot_mixing.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_separation_tab(self):
        """Create separation tab"""
        control_frame = ttk.LabelFrame(self.tab_separation, text="ICA Parameters", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Max Iterations:").pack(side=tk.LEFT, padx=5)
        self.max_iter_var = tk.IntVar(value=200)
        ttk.Entry(control_frame, textvariable=self.max_iter_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Tolerance:").pack(side=tk.LEFT, padx=5)
        self.tol_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(control_frame, textvariable=self.tol_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Run FastICA", 
                  command=self.run_fastica).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(control_frame, text="Save Separated Sources", 
                  command=self.save_separated).pack(side=tk.LEFT, padx=5)
        
        self.ica_status_label = ttk.Label(control_frame, text="")
        self.ica_status_label.pack(side=tk.LEFT, padx=10)
        
        self.plot_separation = PlotCanvas(self.tab_separation, figsize=(12, 8))
        self.plot_separation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_nmf_tab(self):
        """Create NMF tab"""
        control_frame = ttk.LabelFrame(self.tab_nmf, text="NMF Parameters", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Max Iterations:").pack(side=tk.LEFT, padx=5)
        self.nmf_max_iter_var = tk.IntVar(value=200)
        ttk.Entry(control_frame, textvariable=self.nmf_max_iter_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Tolerance:").pack(side=tk.LEFT, padx=5)
        self.nmf_tol_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(control_frame, textvariable=self.nmf_tol_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Run NMF", 
                  command=self.run_nmf).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(control_frame, text="Compare ICA vs NMF", 
                  command=self.compare_ica_nmf).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save NMF Sources", 
                  command=self.save_nmf_separated).pack(side=tk.LEFT, padx=5)
        
        self.nmf_status_label = ttk.Label(control_frame, text="")
        self.nmf_status_label.pack(side=tk.LEFT, padx=10)
        
        self.plot_nmf = PlotCanvas(self.tab_nmf, figsize=(12, 8))
        self.plot_nmf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_recognition_tab(self):
        """Create recognition tab"""
        control_frame = ttk.LabelFrame(self.tab_recognition, text="DTW Recognition", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load Template Dataset", 
                  command=self.load_templates).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Recognize Separated Sources", 
                  command=self.recognize_sources).pack(side=tk.LEFT, padx=5)
        
        self.recognition_status = ttk.Label(control_frame, text="No templates loaded")
        self.recognition_status.pack(side=tk.LEFT, padx=10)
        
        results_frame = ttk.LabelFrame(self.tab_recognition, text="Recognition Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.recognition_text = tk.Text(results_frame, height=20, width=80)
        self.recognition_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, command=self.recognition_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.recognition_text.config(yscrollcommand=scrollbar.set)
    
    def _create_evaluation_tab(self):
        """Create evaluation tab"""
        control_frame = ttk.LabelFrame(self.tab_evaluation, text="Evaluation Metrics", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Compute Metrics", 
                  command=self.compute_metrics).pack(side=tk.LEFT, padx=5)
        
        results_frame = ttk.LabelFrame(self.tab_evaluation, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.metrics_text = tk.Text(results_frame, height=30, width=100, font=('Courier', 10))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def select_audio_files(self):
        """Select audio files for mixing"""
        files = filedialog.askopenfilenames(
            title="Select 4-5 WAV files",
            filetypes=[("WAV files", "*.wav")]
        )
        
        if len(files) < 4 or len(files) > 5:
            messagebox.showwarning("Warning", "Please select 4-5 audio files")
            return
        
        self.audio_files = list(files)
        self.audio_data = []
        self.sample_rates = []
        
        for filepath in self.audio_files:
            data, sr = load_wav(filepath)
            self.audio_data.append(data)
            self.sample_rates.append(sr)
        
        filenames = [os.path.basename(f) for f in self.audio_files]
        self.file_list_label.config(text=f"{len(files)} files: {', '.join(filenames[:3])}...")
        
        messagebox.showinfo("Success", f"Loaded {len(files)} audio files")
    
    def generate_mixtures(self):
        """Generate mixture signals"""
        if len(self.audio_data) == 0:
            messagebox.showwarning("Warning", "Please select audio files first")
            return
        
        self.mixtures, self.mixing_matrix = create_mixtures(self.audio_data)
        
        fig = self.plot_mixing.get_figure()
        fig.clear()
        
        n_sources = len(self.audio_data)
        axes = fig.subplots(n_sources + 1, 1)
        
        for i in range(n_sources):
            plot_waveform(self.mixtures[i], self.sample_rates[0], 
                         f"Mixture {i+1}", ax=axes[i])
        
        plot_mixing_matrix(self.mixing_matrix, ax=axes[-1])
        
        self.plot_mixing.update_plot()
        
        messagebox.showinfo("Success", f"Generated {len(self.mixtures)} mixtures")
    
    def save_mixtures(self):
        """Save mixture signals to files"""
        if self.mixtures is None:
            messagebox.showwarning("Warning", "Please generate mixtures first")
            return
        
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        for i, mixture in enumerate(self.mixtures):
            filename = os.path.join(output_dir, f"mixture_{i+1}.wav")
            save_wav(filename, mixture, self.sample_rates[0])
        
        messagebox.showinfo("Success", f"Saved {len(self.mixtures)} mixtures to {output_dir}")
    
    def run_fastica(self):
        """Run FastICA algorithm"""
        if self.mixtures is None:
            messagebox.showwarning("Warning", "Please generate mixtures first")
            return
        
        self.ica_status_label.config(text="Running FastICA...")
        self.root.update()
        
        self.ica_model = FastICA(
            n_components=len(self.mixtures),
            max_iter=self.max_iter_var.get(),
            tol=self.tol_var.get(),
            random_state=42
        )
        
        self.separated_sources = self.ica_model.fit_transform(self.mixtures)
        
        aligned_sources, permutation, correlations = permutation_solver(
            np.array(self.audio_data), 
            self.separated_sources
        )
        self.separated_sources = aligned_sources
        
        fig = self.plot_separation.get_figure()
        fig.clear()
        
        plot_comparison(
            self.audio_data,
            self.mixtures,
            self.separated_sources,
            self.sample_rates[0],
            titles=[f"Source {i+1}" for i in range(len(self.audio_data))]
        )
        
        self.plot_separation.update_plot()
        
        self.ica_status_label.config(
            text=f"FastICA completed in {self.ica_model.n_iter} iterations"
        )
        
        messagebox.showinfo("Success", "Source separation completed!")
    
    def save_separated(self):
        """Save separated sources"""
        if self.separated_sources is None:
            messagebox.showwarning("Warning", "Please run FastICA first")
            return
        
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        for i, source in enumerate(self.separated_sources):
            filename = os.path.join(output_dir, f"separated_source_{i+1}.wav")
            save_wav(filename, source, self.sample_rates[0])
        
        messagebox.showinfo("Success", f"Saved {len(self.separated_sources)} separated sources")
    
    def run_nmf(self):
        """Run NMF algorithm"""
        if self.mixtures is None:
            messagebox.showwarning("Warning", "Please generate mixtures first")
            return
        
        self.nmf_status_label.config(text="Running NMF...")
        self.root.update()
        
        self.nmf_model = NMF(
            n_components=len(self.mixtures),
            max_iter=self.nmf_max_iter_var.get(),
            tol=self.nmf_tol_var.get(),
            random_state=42
        )
        
        # Run NMF on first mixture
        self.nmf_separated = self.nmf_model.separate_sources(
            self.mixtures[0], 
            self.sample_rates[0]
        )
        self.nmf_separated = np.array(self.nmf_separated)
        
        # Align with original sources
        sources_padded = pad_signals(self.audio_data)
        aligned_sources, permutation, correlations = permutation_solver(
            sources_padded,
            self.nmf_separated
        )
        self.nmf_separated = aligned_sources
        
        # Visualize
        fig = self.plot_nmf.get_figure()
        fig.clear()
        
        plot_comparison(
            self.audio_data,
            self.mixtures,
            self.nmf_separated,
            self.sample_rates[0],
            titles=[f"Source {i+1}" for i in range(len(self.audio_data))]
        )
        
        self.plot_nmf.update_plot()
        
        self.nmf_status_label.config(
            text=f"NMF completed in {self.nmf_model.n_iter} iterations"
        )
        
        messagebox.showinfo("Success", "NMF separation completed!")
    
    def save_nmf_separated(self):
        """Save NMF separated sources"""
        if self.nmf_separated is None:
            messagebox.showwarning("Warning", "Please run NMF first")
            return
        
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        for i, source in enumerate(self.nmf_separated):
            filename = os.path.join(output_dir, f"nmf_separated_source_{i+1}.wav")
            save_wav(filename, source, self.sample_rates[0])
        
        messagebox.showinfo("Success", f"Saved {len(self.nmf_separated)} NMF separated sources")
    
    def compare_ica_nmf(self):
        """Compare ICA and NMF results"""
        if self.separated_sources is None:
            messagebox.showwarning("Warning", "Please run ICA first")
            return
        
        if self.nmf_separated is None:
            messagebox.showwarning("Warning", "Please run NMF first")
            return
        
        # Create comparison window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("ICA vs NMF Comparison")
        compare_window.geometry("800x600")
        
        # Calculate metrics
        sources_padded = pad_signals(self.audio_data)
        
        ica_snr = [snr(sources_padded[i], self.separated_sources[i]) for i in range(len(sources_padded))]
        ica_sdr = [sdr(sources_padded[i], self.separated_sources[i]) for i in range(len(sources_padded))]
        
        nmf_snr = [snr(sources_padded[i], self.nmf_separated[i]) for i in range(len(sources_padded))]
        nmf_sdr = [sdr(sources_padded[i], self.nmf_separated[i]) for i in range(len(sources_padded))]
        
        # Display results
        text_widget = tk.Text(compare_window, font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, "=" * 80 + "\n")
        text_widget.insert(tk.END, "ICA vs NMF Comparison\n")
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        text_widget.insert(tk.END, f"{'Source':<10} {'ICA SNR (dB)':<15} {'NMF SNR (dB)':<15} {'ICA SDR (dB)':<15} {'NMF SDR (dB)':<15}\n")
        text_widget.insert(tk.END, "=" * 80 + "\n")
        
        for i in range(len(sources_padded)):
            text_widget.insert(tk.END, 
                f"{i+1:<10} {ica_snr[i]:<15.2f} {nmf_snr[i]:<15.2f} {ica_sdr[i]:<15.2f} {nmf_sdr[i]:<15.2f}\n"
            )
        
        text_widget.insert(tk.END, "=" * 80 + "\n")
        text_widget.insert(tk.END, 
            f"{'Average':<10} {np.mean(ica_snr):<15.2f} {np.mean(nmf_snr):<15.2f} "
            f"{np.mean(ica_sdr):<15.2f} {np.mean(nmf_sdr):<15.2f}\n"
        )
        text_widget.insert(tk.END, "=" * 80 + "\n\n")
        
        # Conclusion
        if np.mean(nmf_snr) > np.mean(ica_snr):
            diff = np.mean(nmf_snr) - np.mean(ica_snr)
            text_widget.insert(tk.END, f"✓ NMF performs better by {diff:.2f} dB SNR\n")
        else:
            diff = np.mean(ica_snr) - np.mean(nmf_snr)
            text_widget.insert(tk.END, f"✓ ICA performs better by {diff:.2f} dB SNR\n")
        
        text_widget.config(state=tk.DISABLED)
    
    def load_templates(self):
        """Load template dataset for DTW"""
        template_dir = filedialog.askdirectory(title="Select template directory")
        if not template_dir:
            return
        
        wav_files = [f for f in os.listdir(template_dir) if f.endswith('.wav')]
        
        if len(wav_files) == 0:
            messagebox.showwarning("Warning", "No WAV files found in directory")
            return
        
        self.templates_mfcc = []
        self.templates_labels = []
        
        for wav_file in wav_files:
            filepath = os.path.join(template_dir, wav_file)
            data, sr = load_wav(filepath)
            
            mfcc_features = mfcc(data, sr)
            
            label = wav_file.replace('.wav', '').replace('digit_', '').replace('letter_', '')
            
            self.templates_mfcc.append(mfcc_features.T)
            self.templates_labels.append(label)
        
        self.dtw_classifier = DTWClassifier()
        self.dtw_classifier.fit(self.templates_mfcc, self.templates_labels)
        
        self.recognition_status.config(text=f"Loaded {len(self.templates_labels)} templates")
        messagebox.showinfo("Success", f"Loaded {len(self.templates_labels)} templates")
    
    def recognize_sources(self):
        """Recognize separated sources using DTW"""
        if self.separated_sources is None:
            messagebox.showwarning("Warning", "Please run FastICA first")
            return
        
        if self.dtw_classifier is None:
            messagebox.showwarning("Warning", "Please load templates first")
            return
        
        self.recognition_text.delete(1.0, tk.END)
        self.recognition_text.insert(tk.END, "=== DTW Recognition Results ===\n\n")
        
        for i, source in enumerate(self.separated_sources):
            mfcc_features = mfcc(source, self.sample_rates[0])
            
            label, distance = self.dtw_classifier.predict_single(mfcc_features.T)
            
            self.recognition_text.insert(tk.END, 
                f"Separated Source {i+1}:\n"
                f"  Predicted Label: {label}\n"
                f"  DTW Distance: {distance:.2f}\n\n"
            )
        
        messagebox.showinfo("Success", "Recognition completed!")
    
    def compute_metrics(self):
        """Compute SNR and SDR metrics"""
        if self.separated_sources is None:
            messagebox.showwarning("Warning", "Please run FastICA first")
            return
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "=== Evaluation Metrics ===\n\n")
        
        snr_values = []
        sdr_values = []
        
        for i in range(len(self.audio_data)):
            snr_val = snr(self.audio_data[i], self.separated_sources[i])
            sdr_val = sdr(self.audio_data[i], self.separated_sources[i])
            
            snr_values.append(snr_val)
            sdr_values.append(sdr_val)
            
            self.metrics_text.insert(tk.END, 
                f"Source {i+1}:\n"
                f"  SNR: {snr_val:.2f} dB\n"
                f"  SDR: {sdr_val:.2f} dB\n\n"
            )
        
        avg_snr = np.mean(snr_values)
        avg_sdr = np.mean(sdr_values)
        
        self.metrics_text.insert(tk.END, 
            f"\n{'='*40}\n"
            f"Average Metrics:\n"
            f"  Average SNR: {avg_snr:.2f} dB\n"
            f"  Average SDR: {avg_sdr:.2f} dB\n"
        )
        
        messagebox.showinfo("Success", "Metrics computed!")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AudioSeparationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
