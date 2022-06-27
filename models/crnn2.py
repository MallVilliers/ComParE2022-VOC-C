import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH
import librosa
import numpy as np


class CRNN2(nn.Module):
    def __init__(self, feature_type, mel, n_mels=40):
        super(CRNN2, self).__init__()
        self.feature_type = feature_type
        self.mel = mel

        # fbank特征
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=1323, hop_length=441, \
                                                 f_min=0, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mels)
        )

        in_channels = FEATURES_LENGTH[feature_type] if feature_type else n_mels
        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(32 * n_mels // 2 // 2, num_layers=2, hidden_size=128)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, 6)
        )

        print(f'CRNN2 Model init completed, feature_type is {feature_type}, spectrogram is {mel}')

    def forward(self, x1, x2, aug=False):
        if self.feature_type:
            x = x2.unsqueeze(-1)
        else:
            x = x1
            if self.mel == 'fbank':
                with torch.no_grad():
                    x = self.torchfbank(x1) + 1e-6
                    x1 = x1.log()
                    x = x1 - torch.mean(x1, dim=-1, keepdim=True)  # [32, 128, 103]
                    x = torch.FloatTensor(librosa.feature.mfcc(y=np.array(x1), sr=44100, n_mfcc=40))
        x = x.unsqueeze(1)  # [32, 1, 80, 103]
        x = self.conv1(x)  # [32, 16, 40, 51]
        x = self.conv2(x)  # [32, 32, 20, 25]

        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(2, 0, 1)     # [25, 32, 32 * 20]
        output, (hc, nc) = self.lstm(x)  # [25, 32, 128], [2, 32, 128]

        hc = hc.permute(1, 0, 2)    # [32, 2, 128]
        hc = hc.contiguous().view(hc.shape[0], -1)   # [32, 256]
        y = self.fc(hc)   # [32, 6]

        return y


if __name__ == '__main__':
    data = torch.randn(32, 44100 + 882)
    # data = torch.randn(32, 1536)
    m = CRNN2(None)
    o = m.forward(data, data)
    print(o.size())
