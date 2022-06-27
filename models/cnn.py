import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH


class CNN(nn.Module):
    def __init__(self, feature_type, mel):
        super(CNN, self).__init__()
        self.mel = mel

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=FEATURES_LENGTH[feature_type], out_channels=512, kernel_size=5, padding=2),
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

        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 6),
            nn.Softmax(-1)
        )

        print(f'CNN Model init completed, feature_type is {feature_type}, spectrogram is {mel}')

    def forward(self, x1, x, aug=False):

        x = x.unsqueeze(-1)     # [32, 1536, 1]
        x = self.conv1(x)   # [32, 512, 1]
        x = self.conv2(x)   # [32, 256, 1]
        x = x.squeeze(-1)   # [32, 256]
        x = self.fc1(x)     # [32, 64]
        y = self.fc2(x)     # [32, 6]

        return y


if __name__ == '__main__':
    data = torch.randn(32, 1536)
    m = CNN('audeep')
    o = m.forward(data, data)
    print(o.size())