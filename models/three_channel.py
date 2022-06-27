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


class ThreeChannel(nn.Module):

    def __init__(self, feature_type, n_mels=128):
        super(ThreeChannel, self).__init__()
        self.feature_type = feature_type
        self.feature_dim = FEATURES_LENGTH[feature_type] if feature_type else 64  # 获取指定特征的维度
        self.specaug = FbankAug()  # 数据增强（加mask）

        # fbank特征
        self.torchfbank1 = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=1323, hop_length=441, \
                                                 f_min=20, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mels),
        )

        self.torchfbank2 = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, win_length=882, hop_length=441, \
                                                 f_min=20, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mels),
        )

        # 多维度卷积
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding=(1, 0))
        self.conv1b = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.bn1a = nn.BatchNorm2d(32)
        self.bn1b = nn.BatchNorm2d(32)

        # 两层卷积
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(2),
        )

        # lstm + fc
        dim_i = n_mels // 2 // 2
        self.lstm = nn.LSTM(dim_i, 128, bidirectional=False)
        self.fc1 = nn.Linear(in_features=128, out_features=64)

        # 通道2的fc
        # self.fc2 = nn.Sequential(
        #     nn.Linear(in_features=self.feature_dim, out_features=1024),
        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.Linear(in_features=256, out_features=64),
        # )

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
        in_features = 256 if feature_type else 128 + 64
        self.classification = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=6),
            nn.Softmax(dim=-1)
        )

        print(f'ThreeChannel Model init completed, feature_type is {feature_type}')

    def forward(self, x1, x2, aug=True):
        """

        :param x1: 原始语音
        :param x2: 指定特征
        :param aug: 是否做mask（被注释了，打开注释的话，则训练加mask,测试不加）
        :return:
        """
        with torch.no_grad():
            x1_1 = self.torchfbank1(x1) + 1e-6
            x1_1 = x1_1.log()
            x1_1 = x1_1 - torch.mean(x1_1, dim=-1, keepdim=True)
            # if aug == True:
            #     x = self.specaug(x)
            x1_2 = self.torchfbank2(x1) + 1e-6
            x1_2 = x1_2.log()
            x1_2 = x1_2 - torch.mean(x1_2, dim=-1, keepdim=True)

        # 通道1
        x1_1 = x1_1.unsqueeze(1)
        xa_1 = self.conv1a(x1_1)  # (64, 32, 80, 103)
        xb_1 = self.conv1b(x1_1)
        x1_1 = torch.cat((xa_1, xb_1), 1)  # (64, 64, 80, 103)
        x1_1 = self.conv2(x1_1)  # (64, 1, 20, 25)
        x1_1 = x1_1.squeeze(1).permute(2, 0, 1)  # (25, 64, 20)
        output1_1, (hn1_1, cn1_1) = self.lstm(x1_1)  # output: (25, 64, 128), hn: (1, 64, 128)
        x1_1 = hn1_1.squeeze(0)     # (64, 128)

        # 通道2
        x1_2 = x1_2.unsqueeze(1)
        xa_2 = self.conv1a(x1_2)  # (64, 32, 80, 103)
        xb_2 = self.conv1b(x1_2)
        x1_2 = torch.cat((xa_2, xb_2), 1)  # (64, 64, 80, 103)
        x1_2 = self.conv2(x1_2)  # (64, 1, 20, 25)
        x1_2 = x1_2.squeeze(1).permute(2, 0, 1)  # (25, 64, 20)
        output1_1, (hn1_1, cn1_1) = self.lstm(x1_2)  # output: (25, 64, 128), hn: (1, 64, 128)
        x1_2 = hn1_1.squeeze(0)
        x1_2 = self.fc1(x1_2)  # (64, 64)

        if self.feature_type:
            # 通道3
            # x2 = self.fc2(x2)  # (64, 64)
            x2 = x2.unsqueeze(-1)
            x2 = self.conv3(x2)
            x2 = x2.squeeze(-1)
            x2 = self.fc3(x2)

            # 拼接三通道
            x = torch.cat((x1_1, x1_2, x2), 1)  # (64, 128 + 64 + 64)
        else:
            x = torch.cat((x1_1, x1_2), 1)   # (64, 128 + 64)

        # 分类
        y = self.classification(x)  # (64, 6)

        return y


if __name__ == '__main__':
    x1 = torch.randn(64, 44100)
    # x1 = torch.randn(64, 16000)
    x2 = torch.randn(64, 6374)
    # feature_type = 'opensmile'
    feature_type = None
    m = ThreeChannel(feature_type)
    y = m.forward(x1, x2)
    print(y.size())
