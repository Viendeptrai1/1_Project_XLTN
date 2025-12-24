"""
Audio Player Widget for Tkinter
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import threading

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio playback disabled.")


class AudioPlayer(ttk.Frame):
    """
    Audio player widget with play/pause/stop controls
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.audio_data = None
        self.sample_rate = 16000
        self.is_playing = False
        self.play_thread = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create player controls"""
        self.play_btn = ttk.Button(self, text="▶ Play", command=self.play)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(self, text="■ Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.status_label = ttk.Label(self, text="No audio loaded")
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def load_audio(self, audio_data, sample_rate):
        """Load audio data for playback"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        duration = len(audio_data) / sample_rate
        self.status_label.config(text=f"Ready ({duration:.2f}s)")
        self.play_btn.config(state=tk.NORMAL)
    
    def play(self):
        """Play audio"""
        if not AUDIO_AVAILABLE:
            self.status_label.config(text="Audio playback not available")
            return
        
        if self.audio_data is None:
            return
        
        if self.is_playing:
            return
        
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Playing...")
        
        self.play_thread = threading.Thread(target=self._play_audio)
        self.play_thread.start()
    
    def _play_audio(self):
        """Internal audio playback method"""
        try:
            sd.play(self.audio_data, self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self.is_playing = False
            self.after(0, self._playback_finished)
    
    def _playback_finished(self):
        """Called when playback finishes"""
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Finished")
    
    def stop(self):
        """Stop audio playback"""
        if AUDIO_AVAILABLE:
            sd.stop()
        self.is_playing = False
        self._playback_finished()
