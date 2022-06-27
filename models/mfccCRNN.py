import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import soundfile
import librosa
import numpy as np
import random
from scipy import signal
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=16, padding=1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Sequential(nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),)
        self.pool2=nn.MaxPool2d(2)
        self.fc=nn.Sequential(nn.Linear(256,64),
                              nn.Linear(64,6)
                              )
        self.lstm=nn.LSTM(320,num_layers=2,hidden_size =128)
    def forward(self,x):
        x = librosa.feature.mfcc(y=x, sr=44100, n_mfcc=40)
        # f,t,x=signal.spectrogram(x,44100,window='hamming',nfft=2048,noverlap=882,nperseg=1323)
        x=torch.FloatTensor(x)
        x=x.unsqueeze(1)
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=x.reshape(x.shape[0],-1,22).permute(2,0,1)
        print(x.shape)
        x,(h,c)=self.lstm(x)
        print(h.shape)
        h=h.permute(1,0,2)
        print(h.shape)
        # h=h.reshape(h.shape[0],-1)
        h=torch.cat((h[:,0],h[:,1]),dim=1)
        print(h.shape)
        out=self.fc(h)
        return out