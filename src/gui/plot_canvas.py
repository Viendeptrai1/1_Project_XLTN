"""
Plot Canvas Widget - Embed Matplotlib in Tkinter
"""

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotCanvas(tk.Frame):
    """
    Canvas for embedding matplotlib plots in Tkinter
    """
    
    def __init__(self, parent, figsize=(10, 6), **kwargs):
        super().__init__(parent, **kwargs)
        
        self.figure = Figure(figsize=figsize, dpi=100)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def clear(self):
        """Clear all plots"""
        self.figure.clear()
        self.canvas.draw()
    
    def get_figure(self):
        """Get matplotlib figure"""
        return self.figure
    
    def update_plot(self):
        """Refresh the canvas"""
        self.canvas.draw()
