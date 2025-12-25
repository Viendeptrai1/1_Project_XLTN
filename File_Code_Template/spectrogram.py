import tkinter as tk
import tkinter.filedialog as fd
import soundfile as sf
import numpy as np

class App(tk.Tk):
    def	__init__(self):
        super().__init__()
        self.title("Spectrogram")
        
        #Declare variables and initialise them
        self.data = None

        self.cvs_figure = tk.Canvas(self, width=600, height=600, relief = tk.SUNKEN, border = 1)
        lblf_upper = tk.LabelFrame(self)
        
        btn_open = tk.Button(lblf_upper, text="Open", width = 10, command=self.btn_open_clicked)
        btn_cut = tk.Button(lblf_upper, text="Cut", width = 10, command=self.btn_cut_clicked)
        btn_spectrogram = tk.Button(lblf_upper, text="Spectrogram", width = 10, command=self.btn_spectrogram_clicked)

        btn_open.grid(row=0, padx=5, pady=5)
        btn_cut.grid(row=1, padx=5, pady=5)
        btn_spectrogram.grid(row=2, padx=5, pady=5)
        
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
                
                self.cvs_figure.create_line(i, 450 - y1, i + 1, 450 - y2)
        
        
    def btn_cut_clicked(self):
        index_bat_dau = 8
        bat_dau = index_bat_dau * 600
        index_ket_thuc = 15
        ket_thuc = index_ket_thuc * 600
        data_temp = self.data[bat_dau:ket_thuc]
        
        L = len(data_temp)
        N = L // 600
        print('L = ', L)
        print('N = ', N)
        
        self.cvs_figure.delete(tk.ALL)
        for i in range(0, 599):
            x1 = int(data_temp[i*N])
            y1 = int((x1 + 32768) * 300 / 65535) - 150
            
            x2 = int(data_temp[(i + 1) * N])
            y2 = int((x2 + 32768) * 300 / 65535) - 150
            
            self.cvs_figure.create_line(i, 450 - y1, i + 1, 450 - y2)
           
            
    def btn_spectrogram_clicked(self):
        index_bat_dau = 8
        bat_dau = index_bat_dau * 600
        index_ket_thuc = 15
        ket_thuc = index_ket_thuc * 600
        
        data_temp = self.data[bat_dau:ket_thuc]
        data_temp = data_temp.astype('float32')
        data_temp = data_temp / 32768
        L = len(data_temp)
        N = L // 600
        pad_zeros = np.zeros((112,), dtype='float32')
        yc = 300
        for x in range(0, 600):
            x1 = x*N
            x2 = x*N + 400
            frame = data_temp[x1:x2]
            y = np.hstack((frame, pad_zeros))
            Y = np.fft.fft(y, 512)
            scale = 1.0
            S = scale * np.sqrt(Y.real**2 + Y.imag**2)
            S = np.clip(S, 0.001, 400)
            S = 20 * np.log10(S)
            dark = -(S - 52) / 112 * 255
            dark = dark[:257]
            dark = dark.astype(np.int32)
            for k in range(0, 257):
                mau = '#%02x%02x%02x' % (dark[k], dark[k], dark[k])
                self.cvs_figure.create_line(x, yc - k, x, yc - (k + 1), fill = mau)
            
        
           
if __name__ == "__main__":
    app = App()
    app.mainloop()
    
    