import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import soundfile
import librosa
import numpy as np
import random
from scipy import signal
class DSCNN(nn.Module):
    def __init__(self):
        super(DSCNN,self).__init__()
        self.ds_conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3, stride=2, padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Dropout(),
        )
        self.ds_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(),
            )
        self.pool=nn.AvgPool2d(2)
        self.fc=nn.Sequential(
            nn.Linear(1760,6),
        )
    def forward(self,x):
        x = librosa.feature.mfcc(y=x, sr=44100, n_mfcc=40)
        x = torch.FloatTensor(x)
        x = x.unsqueeze(1)
        x=self.ds_conv1(x)
        x=self.ds_conv2(x)
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        return x
# if __name__ == '__main__':
#     audio, sr = soundfile.read('dist/wav/train_0002.wav')
#     length = sr * 1 + 882  # sr * 1: 1s的语音长度，882：是30ms一帧，10ms一帧移，最后一帧会少20ms的语音长度
#     if audio.shape[0] <= length:
#         shortage = length - audio.shape[0]
#         audio = np.pad(audio, (0, shortage), 'wrap')
#     start_frame = np.int64(random.random() * (audio.shape[0] - length))
#     audio = audio[start_frame: start_frame + length]
#     audio /= np.std(audio)
#     audio -= np.mean(audio)
#     r2 = DSCNN()
#     out = r2(audio)
#     print(out.shape)
