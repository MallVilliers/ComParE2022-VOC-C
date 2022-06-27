import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile

# 情绪-label对照表
EMOTION_MAPPING = {
    'achievement': 0,
    'anger': 1,
    'fear': 2,
    'pain': 3,
    'pleasure': 4,
    'surprise': 5
}

# 特征长度
FEATURES_LENGTH = {
    'audeep': 1536,
    'deepspectrum': 1664,
    'opensmile': 6374,
    'xbow': 2000
}


def load_feature(feature_type, feature_path):
    """加载已有特征集"""
    sep = ';' if feature_type == 'opensmile' else ','  # 指定分隔符
    feat_df = pd.read_csv(feature_path + feature_type + '/features.csv', sep=sep, quotechar="'")  # 加载特征集

    return feat_df


class TrainLoader(Dataset):

    def __init__(self, feat_df, train_path, wav_path, feature_type, **kwargs):
        super(TrainLoader, self).__init__()
        self.data_list = []  # 用来记录原始语音
        self.data_list_feat = []  # 用来记录指定特征
        self.data_label = []  # 记录标签
        self.feature_type = feature_type
        # 加载label
        label_df = pd.read_csv(train_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'filename' if feature_type == 'audeep' else 'name'  # 特征集第一个字段名
            self.train_df = feat_df[feat_df[name].str.contains('train')].reset_index(drop=True)  # 获取训练特征集

        for index in range(len(label_df)):
            label = EMOTION_MAPPING[label_df.loc[index, 'label']]
            # 记录标签
            self.data_label.append(label)
            # 记录指定特征索引
            if feature_type:
                self.data_list_feat.append(
                    self.train_df[self.train_df[name] == label_df.loc[index, 'filename']].index.item())
            # 记录原始语音文件
            file_name = os.path.join(wav_path, label_df.loc[index, 'filename'])
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # 指定特征
        if self.feature_type:
            feature = self.train_df.iloc[self.data_list_feat[index], 1:].astype(float).values
        else:
            feature = [0.0]

        # 用soundfile加载音频 -> 音频向量， 采样率
        audio, sr = soundfile.read(self.data_list[index])

        length = sr * 1 + 882  # sr * 1: 1s的语音长度，882：是30ms一帧，10ms一帧移，最后一帧会少20ms的语音长度
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame: start_frame + length]
        audio /= np.std(audio)
        audio -= np.mean(audio)

        return torch.FloatTensor(audio), torch.FloatTensor(feature), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
