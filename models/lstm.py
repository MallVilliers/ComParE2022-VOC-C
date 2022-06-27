import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH


class LSTM(nn.Module):
    def __init__(self, feature_type):
        super(LSTM, self).__init__()

        self.lstm = nn.Sequential(
            nn.LSTM(FEATURES_LENGTH[feature_type], 512),
            # nn.Dropout(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 126),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(126, 6),
            nn.Softmax(-1)
        )

        print(f'LSTM Model init completed, feature_type is {feature_type}')

    def forward(self, x2, x, aug=False):
        # 增加一个序列维
        x = x.unsqueeze(0)  # (1, 32, 1536)

        output, (hn, cn) = self.lstm(x)     # ouput: (1, 32, 512), hn: (1, 32, 512)

        x = hn.squeeze(0)  # (32, 512)

        x = self.fc1(x)     # (32, 126)
        y = self.fc2(x)     # (32, 6)

        return y


if __name__ == '__main__':
    data = torch.randn(32, 1536)
    m = LSTM('audeep')
    o = m.forward(data, data)
    print(o.size())
