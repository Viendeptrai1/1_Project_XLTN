from fileinput import filename
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import tkinter.filedialog as fd
import sounddevice as sd
import queue
import soundfile as sf
import threading

class App(tk.Tk):
    def	__init__(self):
        super().__init__()
        self.title("Speech Signal Processing")
        
        #Create a queue to contain the audio data
        self.q = queue.Queue()
        #Declare variables and initialise them
        self.recording = False
        self.file_exists = False
        self.data = None
        self.filename = None
        self.index = -1

        self.cvs_figure = tk.Canvas(self, width=600, height=300, relief = tk.SUNKEN, border = 1)
        lblf_upper = tk.LabelFrame(self)
        lblf_lower = tk.LabelFrame(self)
        
        btn_record = tk.Button(lblf_upper, text="Record", width = 8, command=lambda m=1:self.threading_rec(m))
        btn_stop = tk.Button(lblf_upper, text="Stop", width = 8, command=lambda m=2:self.threading_rec(m))
        btn_play = tk.Button(lblf_upper, text="Play", width = 8, command=lambda m=3:self.threading_rec(m))
        btn_open = tk.Button(lblf_upper, text="Open", width = 8, command=self.btn_open_clicked)
        btn_cut = tk.Button(lblf_upper, text="Cut", width = 8, command=self.btn_cut_clicked)

        btn_open.grid(row=0, padx=5, pady=5)
        btn_cut.grid(row=1, padx=5, pady=5)
        btn_record.grid(row=2, padx=5, pady=5)
        btn_stop.grid(row=3, padx=5, pady=5)
        btn_play.grid(row=4, padx=5, pady=5)
        
        self.factor_zoom = tk.StringVar()
        self.cbo_zoom = ttk.Combobox(lblf_lower, width = 8, textvariable = self.factor_zoom, state = 'disabled')
        self.cbo_zoom.bind("<<ComboboxSelected>>", self.factor_zoom_changed)
        self.cbo_zoom.grid(row=0, padx=5, pady=5)
        
        btn_next = tk.Button(lblf_lower, text="Next", width = 8, command=self.btn_next_clicked)
        btn_next.grid(row=1, padx=5, pady=5)
        btn_prev = tk.Button(lblf_lower, text="Prev", width = 8, command=self.btn_prev_clicked)
        btn_prev.grid(row=2, padx=5, pady=5)
        
        self.cvs_figure.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        lblf_upper.grid(row=0, column=1, padx=5, pady=7, sticky = tk.N)
        lblf_lower.grid(row=1, column=1, padx=5, pady=7, sticky = tk.S)
        
        self.cvs_figure.bind("<Button-1>", self.xu_ly_mouse)
        
    def xu_ly_mouse(self, event):
        x = event.x
    
    #Fit data into queue
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())
        
    #Functions to play, stop and record audio
    #The recording is done as a thread to prevent it being the main process
    def threading_rec(self, x):
        if x == 1:
            #If recording is selected, then the thread is activated
            t1=threading.Thread(target= self.record_audio)
            t1.start()
        elif x == 2:
            #To stop, set the flag to false
            self.recording = False
            messagebox.showinfo(title="Recording", message="Finished")
            self.data, fs = sf.read("trial.wav", dtype='int16')
            L = len(self.data)
            N = L // 600
            self.cbo_zoom['state'] = 'readonly'
            lst_values = []
            for i in range(1, (N + 1)):
                s = '%10d' % i
                lst_values.append(str(i))
            self.cbo_zoom['values'] = lst_values
            
            self.cvs_figure.delete(tk.ALL)
            for i in range(0, 599):
                x1 = int(self.data[i*N])
                y1 = int((x1 + 32768) * 300 / 65535) - 150
                
                x2 = int(self.data[(i + 1) * N])
                y2 = int((x2 + 32768) * 300 / 65535) - 150
                
                self.cvs_figure.create_line(i, 150 - y1, i + 1, 150 - y2)

        elif x == 3:
            #To play a recording, it must exist.
            if self.file_exists:
                #Read the recording if it exists and play it
                data, fs = sf.read("trial.wav", dtype='float32') 
                sd.play(data,fs)
                sd.wait()
            elif self.filename is not None:
                #Read the recording if it exists and play it
                data, fs = sf.read(self.filename, dtype='float32') 
                sd.play(data,fs)
                sd.wait()
            else:
                #Display and error if none is found
                messagebox.showerror(title="Error", message="Record something to play")

    #Recording function
    def record_audio(self):
        #Set to True to record
        self.recording = True
        self.file_exists = False
        #Create a file to save the audio
        messagebox.showinfo(title="Recording Speech", message="Speak into the mic")
        with sf.SoundFile("trial.wav", mode='w', samplerate=16000,
                            channels=1) as file:
        #Create an input stream to record audio without a preset time
                with sd.InputStream(samplerate=16000, channels=1, callback=self.callback):
                    while self.recording == True:
                        #Set the variable to True to allow playing the audio later
                        self.file_exists =True
                        #write into file
                        file.write(self.q.get())

    def btn_zoom_clicked(self):
        self.cvs_figure.delete(tk.ALL)
        i = self.index
        for x in range(0, 599):
            a1 = int(self.data[i * 600 + x])
            y1 = int((a1 + 32768) * 300 / 65535) - 150
            a2 = int(self.data[i * 600 + x + 1])
            y2 = int((a2 + 32768) * 300 / 65535) - 150
            self.cvs_figure.create_line(x, 150 - y1, x + 1, 150 - y2)
            
    def btn_next_clicked(self):
        factor_zoom = self.factor_zoom.get()
        factor_zoom = int(factor_zoom.strip())
        data_temp = self.data[::int(factor_zoom)]
        L = len(data_temp)
        self.cvs_figure.delete(tk.ALL)
        self.index += 1
        i = self.index
        print('index = ', i)
        for x in range(0, 599):
            a1 = int(data_temp[i * 600 + x])
            y1 = int((a1 + 32768) * 300 / 65535) - 150
            a2 = int(data_temp[i * 600 + x + 1])
            y2 = int((a2 + 32768) * 300 / 65535) - 150
            self.cvs_figure.create_line(x, 150 - y1, x + 1, 150 - y2)
            
    def btn_prev_clicked(self):
        factor_zoom = self.factor_zoom.get()
        factor_zoom = int(factor_zoom.strip())
        data_temp = self.data[::int(factor_zoom)]
        L = len(data_temp)
        self.cvs_figure.delete(tk.ALL)
        self.index -= 1
        i = self.index
        for x in range(0, 599):
            a1 = int(data_temp[i * 600 + x])
            y1 = int((a1 + 32768) * 300 / 65535) - 150
            a2 = int(data_temp[i * 600 + x + 1])
            y2 = int((a2 + 32768) * 300 / 65535) - 150
            self.cvs_figure.create_line(x, 150 - y1, x + 1, 150 - y2)
            
    def factor_zoom_changed(self, event):
        factor_zoom = self.factor_zoom.get()
        self.index = -1
        
    def btn_open_clicked(self):
        filetypes = (("Wave files", "*.wav"),)
        self.filename = fd.askopenfilename(title="Open wave files", filetypes=filetypes)
        if filename:
            print(self.filename)
            self.data, fs = sf.read(self.filename, dtype='int16')
            L = len(self.data)
            N = L // 600
            self.cbo_zoom['state'] = 'readonly'
            lst_values = []
            for i in range(1, (N + 1)):
                s = '%10d' % i
                lst_values.append(str(i))
            self.cbo_zoom['values'] = lst_values
            
            self.cvs_figure.delete(tk.ALL)
            for i in range(0, 599):
                x1 = int(self.data[i*N])
                y1 = int((x1 + 32768) * 300 / 65535) - 150
                
                x2 = int(self.data[(i + 1) * N])
                y2 = int((x2 + 32768) * 300 / 65535) - 150
                
                self.cvs_figure.create_line(i, 150 - y1, i + 1, 150 - y2)
        
    def btn_cut_clicked(self):
        # data_mix_1 = self.data[20*600:32*600]
        # n = len(data_mix_1)
        # s = ''
        # for i in range(n):
        #     s += '%6d\n' % int(data_mix_1[i])
            
        # f = open('mix_01.txt', 'wt')
        # f.write(s)
        # f.close()
        
        
        # data_mix_2 = self.data[42*600:54*600]
        # n = len(data_mix_2)
        # s = ''
        # for i in range(n):
        #     s += '%6d\n' % int(data_mix_2[i])
            
        # f = open('mix_02.txt', 'wt')
        # f.write(s)
        # f.close()
        
        
        batDau = 39 * 600 + 255
        ketThuc = 42 * 600 + 355
        data_temp = self.data[batDau:ketThuc]
        self.cvs_figure.delete(tk.ALL)
        for x in range(0, 599):
            a1 = int(data_temp[x])
            y1 = int((a1 + 32768) * 300 / 65535) - 150
            a2 = int(data_temp[x + 1])
            y2 = int((a2 + 32768) * 300 / 65535) - 150
            self.cvs_figure.create_line(x, 150 - y1, x + 1, 150 - y2)
            
if __name__ == "__main__":
    app = App()
    app.mainloop()
    
    
# 8 16
# 28 44