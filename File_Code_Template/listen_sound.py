# Listen to these sounds
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio

mixture_0 = read_audio('Code/mixture_0.wav').squeeze()
source1_0 = read_audio('Code/source1_0.wav').squeeze()
source2_0 = read_audio('Code/source2_0.wav').squeeze()

mixture_1 = read_audio('Code/mixture_1.wav').squeeze()
source1_1 = read_audio('Code/source1_1.wav').squeeze()
source2_1 = read_audio('Code/source2_1.wav').squeeze()

mixture_2 = read_audio('Code/mixture_2.wav').squeeze()
source1_2 = read_audio('Code/source1_2.wav').squeeze()
source2_2 = read_audio('Code/source2_2.wav').squeeze()

mixture_3 = read_audio('Code/mixture_3.wav').squeeze()
source1_3 = read_audio('Code/source1_3.wav').squeeze()
source2_3 = read_audio('Code/source2_3.wav').squeeze()

train_mixs = [mixture_0, mixture_1, mixture_2]
train_source1s = [source1_0, source1_1, source1_2]
train_source2s = [source2_0, source2_1, source2_2]

Audio(mixture_0, rate=16000)
Audio(source1_0, rate=16000)
Audio(source2_0, rate=16000)

# Construct the datasets and dataloaders
from torch.utils.data import Dataset, DataLoader

class source_separation_dataset(Dataset):
    def __init__(self, train_mixs, train_source1s, train_source2s):
        self.mixs = train_mixs
        self.train_source1s = train_source1s
        self.train_source2s = train_source2s

    def __len__(self):
        return len(self.mixs)

    def __getitem__(self, idx):
        mix = self.mixs[idx]
        source1 = self.train_source1s[idx]
        source2 = self.train_source2s[idx]
        return mix, source1, source2

train_dataset_audio = source_separation_dataset(train_mixs, train_source1s, train_source2s)
valid_dataset_audio = source_separation_dataset([mixture_2], [source1_2], [source2_2])

train_loader_audio = DataLoader(train_dataset_audio, batch_size=1)
valid_loader_audio = DataLoader(valid_dataset_audio, batch_size=1)

import torch.nn as nn
import torch
from speechbrain.core import SeparationBrain 
from speechbrain.lobes.models.source_separation import simpleseparator
fft_size=1024
model_audio = simpleseparator(fft_size=fft_size, hidden_size=300)


optimizer = lambda x: torch.optim.Adam(x, lr=0.0005)
N_epochs = 100
epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=N_epochs)

separator = SeparationBrain(
        train_loss='si-snr',
        modules={'mdl': model_audio},
        opt_class=optimizer

    )


separator.fit(
            epoch_counter,
            train_loader_audio,
            valid_loader_audio)

class audioseparator(nn.Module):
  def __init__(self, fft_size, hidden_size, num_sources=2, kernel_size=16):
    super(audioseparator, self).__init__()
    self.encoder = nn.Conv1d(in_channels=1, out_channels=fft_size, kernel_size=16, stride=kernel_size//2)

    # MaskNet
    self.rnn = nn.LSTM(input_size=fft_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
    self.output_layer = nn.Linear(in_features=hidden_size*2, out_features=num_sources*(fft_size))

    self.decoder = nn.ConvTranspose1d(in_channels=fft_size, out_channels=1, kernel_size=kernel_size, stride=kernel_size//2)

    self.fft_size = fft_size
    self.hidden_size = hidden_size
    self.num_sources = num_sources

  def forward(self, inp):
    # batch x channels x time
    y = nn.functional.relu(self.encoder(inp.unsqueeze(0)))

    # batch x time x nfft
    y = y.permute(0, 2, 1)

    # batch x time x feature
    rnn_out = self.rnn(y)[0]

    # batch x time x (nfft*num_sources)
    lin_out = self.output_layer(rnn_out)

    # batch x time x nfft x num_sources
    lin_out = lin_out.reshape(lin_out.size(0), lin_out.size(1), -1, self.num_sources)

    # reconstruct in time domain
    sources = []
    all_masks = []
    for n in range(self.num_sources):
      sourcehat_mask = nn.functional.relu(lin_out[:, :, :, n])
      all_masks.append(sourcehat_mask)

      # multiply with mask and magnitude
      T = sourcehat_mask.size(1)
      sourcehat_latent = (sourcehat_mask * y[:, :T, :]).permute(0, 2, 1)

      # reconstruct in time domain with istft
      sourcehat = self.decoder(sourcehat_latent).squeeze(0)
      sources.append(sourcehat)

    return sources, all_masks, y

model_audio = audioseparator(fft_size=fft_size, hidden_size=300, kernel_size=256)
out, _, _ = model_audio.forward(mixture_0.unsqueeze(0))

optimizer = lambda x: torch.optim.Adam(x, lr=0.0002)
N_epochs = 200
epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=N_epochs)

separator = SeparationBrain(
        train_loss='si-snr',
        modules={'mdl': model_audio},
        opt_class=optimizer

    )

separator.fit(
            epoch_counter,
            train_loader_audio,
            valid_loader_audio)

estimated_sources_test, all_masks, mag = model_audio.forward(mixture_3.unsqueeze(0))
estimated_sources_train, all_masks, mag = model_audio.forward(mixture_0.unsqueeze(0))


Audio(estimated_sources_test[0].squeeze().detach(), rate=16000)
Audio(estimated_sources_test[1].squeeze().detach(), rate=16000)
Audio(estimated_sources_train[0].squeeze().detach(), rate=16000)
Audio(estimated_sources_train[1].squeeze().detach(), rate=16000)


# @misc{speechbrainV1,
#   title={Open-Source Conversational AI with {SpeechBrain} 1.0},
#   author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
#   year={2024},
#   eprint={2407.00463},
#   archivePrefix={arXiv},
#   primaryClass={cs.LG},
#   url={https://arxiv.org/abs/2407.00463},
# }


# @misc{speechbrain,
#   title={{SpeechBrain}: A General-Purpose Speech Toolkit},
#   author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
#   year={2021},
#   eprint={2106.04624},
#   archivePrefix={arXiv},
#   primaryClass={eess.AS},
#   note={arXiv:2106.04624}
# }
