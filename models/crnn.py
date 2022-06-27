import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH


class CRNN(nn.Module):
    def __init__(self, feature_type, mel, n_mels=128):
        super(CRNN, self).__init__()
        self.feature_type = feature_type
        self.mel = mel

        # fbank特征
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=1323, hop_length=441, \
                                                 f_min=0, f_max=44100//2, window_fn=torch.hamming_window, n_mels=n_mels)
        )

        in_channels = FEATURES_LENGTH[feature_type] if feature_type else n_mels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.lstm = nn.LSTM(256, 64)

        self.fc1 = nn.Sequential(
            nn.Linear(64, 6),
            nn.Softmax(-1)
        )

        print(f'CRNN Model init completed, feature_type is {feature_type}')

    def forward(self, x1, x2, aug=False):
        if self.feature_type:
            x = x2.unsqueeze(-1)
        else:
            x = x1
            if self.mel == 'fbank':
                with torch.no_grad():
                    x1 = self.torchfbank(x1) + 1e-6
                    x1 = x1.log()
                    x = x1 - torch.mean(x1, dim=-1, keepdim=True)  # [32, 128, 103]

        x = self.conv1(x)  # [32, 512, 103]
        x = self.conv2(x)  # [32, 256, 103]

        x = x.permute(2, 0, 1)  # [103, 32, 256]
        output, (hc, nc) = self.lstm(x)    # [103, 32, 64], [1, 32, 64]

        y = self.fc1(hc.squeeze(0))  # [32, 64]

        return y


if __name__ == '__main__':
    data = torch.randn(32, 44100)
    # data = torch.randn(32, 1536)
    m = CRNN(None)
    o = m.forward(data, data)
    print(o.size())