import sounddevice as sd
import soundfile as sf
from tkinter import *

# Global variable to track recording state
is_recording = False

def Voice_rec():
    global is_recording
    is_recording = True
    b["bg"] = "red"  # Change button color when recording
    stop_button["state"] = "normal"
    
    fs = 48000
    duration = 5
    myrecording = sd.rec(int(duration * fs), 
                         samplerate=fs, channels=2)
    sd.wait()
    
    # Reset states after recording
    is_recording = False
    b["bg"] = "SystemButtonFace"  # Reset to default color
    stop_button["state"] = "disabled"
    
    return sf.write('my_Audio_file.flac', myrecording, fs)

def stop_recording():
    global is_recording
    if is_recording:
        sd.stop()
        is_recording = False
        b["bg"] = "SystemButtonFace"  # Reset to default color
        stop_button["state"] = "disabled"

master = Tk()

Label(master, text=" Voice Recoder : "
     ).grid(row=0, sticky=W, rowspan=5)


b = Button(master, text="Start", command=Voice_rec)
b.grid(row=0, column=2, columnspan=2, rowspan=2,
       padx=5, pady=5)

stop_button = Button(master, text="Stop", command=stop_recording, state="disabled")
stop_button.grid(row=2, column=2, columnspan=2, rowspan=2,
                padx=5, pady=5)

mainloop()