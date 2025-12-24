#!/usr/bin/env python
"""
Run GUI Application
Automatically uses correct Python environment
"""

import sys
import os

# Ensure we're using the right Python with all dependencies
try:
    import tkinter as tk
    from src.gui import AudioSeparationApp
    
    print("Starting GUI Application...")
    root = tk.Tk()
    app = AudioSeparationApp(root)
    root.mainloop()
    
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease make sure you're running with conda python:")
    print("  python main.py")
    print("\nNOT:")
    print("  /usr/bin/python3 main.py")
    sys.exit(1)
except Exception as e:
    print(f"GUI Error: {e}")
    print("\nIf you see tkinter/macOS errors, try running without GUI:")
    print("  python demo.py")
    print("  python demo_nmf.py")
    sys.exit(1)
