import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis

from data_loader import FEATURES_LENGTH


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class TwoChannel(nn.Module):

    def __init__(self, feature_type, mel, n_mel):
        super(TwoChannel, self).__init__()
        self.feature_type = feature_type
        self.mel = mel
        self.feature_dim = FEATURES_LENGTH[feature_type] if feature_type else 64  # 获取指定特征的维度
        self.specaug = FbankAug()  # 数据增强（加mask）

        # fbank特征
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, win_length=882, hop_length=441, \
                                                 f_min=20, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mel),
        )

        # 通道1
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

        self.lstm = nn.LSTM(32 * (n_mel // 2 // 2), num_layers=2, hidden_size=128)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
        )

        # 通道2
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )

        # fc + softmax
        in_features = 128 if feature_type else 64
        self.classification = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=6),
            nn.Softmax(dim=-1)
        )

        print(f'TwoChannel Model init completed, feature_type is {feature_type}, spectrogram is {mel}, '
              f'n_mel is {n_mel}')

    def forward(self, x1, x2, aug=True):
        """

        :param x1: 原始语音
        :param x2: 指定特征
        :param aug: 是否做mask（被注释了，打开注释的话，则训练加mask,测试不加）
        :return:
        """
        # 通道1
        if self.mel is 'fbank':
            with torch.no_grad():
                x1 = self.torchfbank(x1) + 1e-6
                x1 = x1.log()
                x1 = x1 - torch.mean(x1, dim=-1, keepdim=True)
                # if aug == True:
                #     x = self.specaug(x)

        x1 = x1.unsqueeze(1)  # [32, 1, 80, 103]
        x1 = self.conv1(x1)  # [32, 16, 40, 51]
        x1 = self.conv2(x1)  # [32, 32, 20, 25]

        x1 = x1.reshape(x1.shape[0], -1, x1.shape[-1]).permute(2, 0, 1)  # [25, 32, 32 * 20]
        output, (hc, nc) = self.lstm(x1)  # [25, 32, 128], [2, 32, 128]

        hc = hc.permute(1, 0, 2)  # [32, 2, 128]
        hc = hc.contiguous().view(hc.shape[0], -1)  # [32, 256]
        x1 = self.fc(hc)  # [32, 64]

        if self.feature_type:
            # 通道2
            x2 = x2.unsqueeze(-1)
            x2 = self.conv3(x2)
            x2 = x2.squeeze(-1)
            x2 = self.fc3(x2)

            # 拼接倆通道
            x = torch.cat((x1, x2), 1)  # (64, 128)
        else:
            x = x1

        # 分类
        y = self.classification(x)  # (64, 6)

        return y


if __name__ == '__main__':
    # x1 = torch.randn(64, 44100 + 882)
    x1 = torch.randn(64, 40, 88)
    x2 = torch.randn(64, 2000)
    # feature_type = 'opensmile'
    feature_type = 'xbow'
    m = TwoChannel(feature_type, 'mfcc', 40)
    y = m.forward(x1, x2)
    print(y.size())
