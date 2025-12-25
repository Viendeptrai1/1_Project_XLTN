f = open('mix_violin_piano_01.wav', 'rb')
ID = f.read(4)
print('%c%c%c%c' % (ID[0], ID[1], ID[2], ID[3]))

fileSize = f.read(4)
fileSize = fileSize[3]*256**3 + fileSize[2]*256**2 + fileSize[1]*256 + fileSize[0]
print(fileSize)

fileFormatID = f.read(4)
print('%c%c%c%c' % (fileFormatID[0], fileFormatID[1], fileFormatID[2], fileFormatID[3]))

FormatBlocID = f.read(4)
print('%c%c%c%c' % (FormatBlocID[0], FormatBlocID[1], FormatBlocID[2], FormatBlocID[3]))

BlocSize = f.read(4)
BlocSize = BlocSize[3]*256**3 + BlocSize[2]*256**2 + BlocSize[1]*256 + BlocSize[0]
print('%d' % BlocSize)

AudioFormat = f.read(2)
BlocSize = AudioFormat[1]*256 + AudioFormat[0]
print('Format theo chuẩn PCM là %d' % BlocSize)

NbrChannels = f.read(2)
NbrChannels = NbrChannels[1]*256 + NbrChannels[0]
print('Số lượng kênh là %d' % NbrChannels)

Frequency = f.read(4)
Frequency = Frequency[3]*256**3 + Frequency[2]*256**2 + Frequency[1]*256 + Frequency[0]
print('Tần số lấy mẫu là %d' % Frequency)

BytePerSec = f.read(4)
BytePerSec = BytePerSec[3]*256**3 + BytePerSec[2]*256**2 + BytePerSec[1]*256 + BytePerSec[0]
print('Số lượng byte trên một giây là %d' % BytePerSec)

BytePerBloc = f.read(2)
BytePerBloc = BytePerBloc[1]*256 + BytePerBloc[0]
print('Số lượng byte trên một khối là %d' % BytePerBloc)

BitsPerSample = f.read(2)
BitsPerSample = BitsPerSample[1]*256 + BitsPerSample[0]
print('Số bit của một mẫu là %d' % BitsPerSample)

DataBlocID = f.read(4)
print('%c%c%c%c' % (DataBlocID[0], DataBlocID[1], DataBlocID[2], DataBlocID[3]))

DataSize = f.read(4)
DataSize = DataSize[3]*256**3 + DataSize[2]*256**2 + DataSize[1]*256 + DataSize[0]
print('Data Size %d' % DataSize)
SoLuongMau = DataSize // 2
print('Số lượng mẫu là %d' % SoLuongMau)

sample = f.read(2)
sample = f.read(2)
sample = f.read(2)
sample = int.from_bytes(sample, byteorder='little')
f.close()