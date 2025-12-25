import wave

f = wave.open('mix_violin_piano_01.wav')
metadata = f.getparams()
print(metadata)