import tkinter as tk
import tkinter.filedialog as fd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

class App(tk.Tk):
    def	__init__(self):
        super().__init__()
        self.title("Fourier GUI")
        
        #Declare variables and initialise them
        self.data = None

        self.cvs_figure = tk.Canvas(self, width=600, height=300, relief = tk.SUNKEN, border = 1)
        lblf_upper = tk.LabelFrame(self)
        
        btn_open = tk.Button(lblf_upper, text="Open", width = 8, command=self.btn_open_clicked)
        btn_cut = tk.Button(lblf_upper, text="Cut", width = 8, command=self.btn_cut_clicked)
        btn_spectrum = tk.Button(lblf_upper, text="Spectrum", width = 8, command=self.btn_spectrum_clicked)

        btn_open.grid(row=0, padx=5, pady=5)
        btn_cut.grid(row=1, padx=5, pady=5)
        btn_spectrum.grid(row=2, padx=5, pady=5)
        
        # Đưa widget lên lưới
        self.cvs_figure.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        lblf_upper.grid(row=0, column=1, padx=5, pady=7, sticky = tk.N)
    
    
    def btn_open_clicked(self):
        filetypes = (("Wave files", "*.wav"),)
        filename = fd.askopenfilename(title="Open wave files", filetypes=filetypes)
        if filename:
            print(filename)
            self.data, fs = sf.read(filename, dtype='int16')
            L = len(self.data)
            N = L // 600
            
            self.cvs_figure.delete(tk.ALL)
            for i in range(0, 599):
                x1 = int(self.data[i*N])
                y1 = int((x1 + 32768) * 300 / 65535) - 150
                
                x2 = int(self.data[(i + 1) * N])
                y2 = int((x2 + 32768) * 300 / 65535) - 150
                
                self.cvs_figure.create_line(i, 150 - y1, i + 1, 150 - y2)
        
        
    def btn_cut_clicked(self):
        index = 30
        batDau = index * 600
        ketThuc = batDau + 600
        data_temp = self.data[batDau:ketThuc]
        self.cvs_figure.delete(tk.ALL)
        for x in range(0, 599):
            a1 = int(data_temp[x])
            y1 = int((a1 + 32768) * 300 / 65535) - 150
            a2 = int(data_temp[x + 1])
            y2 = int((a2 + 32768) * 300 / 65535) - 150
            self.cvs_figure.create_line(x, 150 - y1, x + 1, 150 - y2)
           
            
    def btn_spectrum_clicked(self):
        index = 8
        batDau = index * 600
        ketThuc = batDau + 600
        x = self.data[batDau:ketThuc]
        N = 16000
        x = x / 32768 
        x = x.astype('float32')
        X = np.fft.fft(x, N)
        S = np.sqrt(X.real**2 + X.imag**2)
        
        # Chuyển sang decibel
        S = 20 * np.log10(S)
        S = S[:N//2+1]
        plt.plot(S)
        plt.show()
        
        
    def btn_spectrum_pre_emphasis_clicked(self):
        index = 8
        batDau = index * 600
        ketThuc = batDau + 600
        x = self.data[batDau:ketThuc]
        N = 16000
        x = x / 32768 
        x = x.astype('float32')
        L = len(x)
        y = np.zeros(L, dtype='float32')
        a = 0.9
        
        for i in range(1, L):
            if i == 0:
                y[i] = x[i] - a * x[i]
            else:
                y[i] = x[i] - a * y[i - 1]
                
        Y = np.fft.fft(y, N)
        S = np.sqrt(Y.real**2 + Y.imag**2)
        
        # Chuyển sang decibel
        S = 20 * np.log10(S)
        S = S[:N//2+1]
        plt.plot(S)
        plt.show()
           
if __name__ == "__main__":
    app = App()
    app.mainloop()
    
    