import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH


class AttentionBase(nn.Module):
    def __init__(self, feature_type, n_mels=80):
        super(AttentionBase, self).__init__()

        # fbank特征
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, win_length=882, hop_length=441, \
                                                 f_min=20, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mels),
        )

        # 通道1
        hidden_size = 128
        self.bilstm = nn.LSTM(n_mels, hidden_size, bidirectional=True)
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.attention1 = nn.Softmax(dim=1)

        # 通道2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(),
            # nn.MaxPool2d(2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 16),
        )

        self.u = nn.Parameter(torch.Tensor(16, 1))
        self.attention2 = nn.Softmax(dim=1)

        # 分类层
        self.classification = nn.Sequential(
            nn.Linear(500 * 16 + 256, 6),
            nn.Softmax(dim=1),
        )

        print(f'AttentionBase Model init completed, feature_type is {feature_type}')

    def forward(self, x, x2, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)  # [32, 80, 103]

        # 通道1
        x1 = x.permute(2, 0, 1)     # [103, 32, 80]
        output, (h, c) = self.bilstm(x1)     # [103, 32, 128], [2, 32, 128]
        h = h.permute(1, 0, 2)    # [32, 2, 128]
        h = h.contiguous().view(h.shape[0], -1)   # [32, 256]
        alpha1 = self.attention1(torch.matmul(h, self.w))    # [32, 1]
        x1 = alpha1 * h   # [32, 256]

        # 通道2
        x2 = x.unsqueeze(1)     # [32, 1, 80, 103]
        x2 = self.conv1(x2)     # [32, 64, 40, 51]
        x2 = self.conv2(x2)     # [32, 64, 20, 25]
        x2 = self.conv3(x2)     # [32, 128, 20, 25]
        x2 = x2.reshape(-1, 128)    # [32 * 20 * 25, 128]
        x2 = self.mlp(x2)   # [32 * 20 * 25, 16]
        e = torch.matmul(F.tanh(x2), self.u)   # [32 * 20 * 25, 1]
        alpha2 = self.attention2(0.3 * e)   # [32 * 20 * 25, 1]
        x2 = alpha2 * x2    # [32 * 20 * 25, 16]
        x2 = x2.reshape(x.shape[0], -1)     # [32, 500 * 16]

        x = torch.cat((x1, x2), dim=1)

        y = self.classification(x)

        return y


if __name__ == '__main__':
    data = torch.randn(32, 44100 + 882)
    m = AttentionBase('audeep')
    o = m.forward(data, data)
    print(o.size())