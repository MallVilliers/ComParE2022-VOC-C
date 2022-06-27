import pandas as pd
import torch
import numpy as np
import librosa
from parselmouth.praat import call
import librosa.display
import soundfile as sf
import pylab
import os,glob
import matplotlib.pyplot as plt
from os.path import basename
from scipy.io import wavfile
from PIL import Image
from tqdm import tqdm
import parselmouth
if __name__ == '__main__':
    audio_base='./dist/wav'
    output_base='./dist/features/spectrum'
    wavs=glob.glob(f'{audio_base}/*.wav')
    os.makedirs(output_base,exist_ok=True)
    train_path='dist/lab/train.csv'
    test_path='dist/lab/test.csv'
    devel_path='dist/lab/devel.csv'
    wav_path1='dist/wav/'
    label_df = pd.read_csv(train_path, sep=',')
    label_df1 = pd.read_csv(test_path, sep=',')
    label_df2 = pd.read_csv(devel_path, sep=',')
    # for index in range(len(label_df)):
    #     label =label_df.loc[index, 'label']
    #     filename=os.path.join(wav_path1, label_df.loc[index, 'filename'])
    #     file=label_df.loc[index, 'filename'].split('\\')[-1].split('.')[0]
    #     # print(file)
    #     print(filename)
        # print(label)
        # y, sr = librosa.load(filename, sr=None)
        # y = librosa.effects.pitch_shift(y, sr, n_steps=-3)
        # sf.write(f'./dist/adjustwav/{file}.wav', y, sr)

    for index in range(len(label_df)):
        label = label_df.loc[index, 'label']
        filename = os.path.join(wav_path1, label_df.loc[index, 'filename'])
        file = label_df.loc[index, 'filename'].split('\\')[-1].split('.')[0]
        print(filename)
        # sound = parselmouth.Sound(filename)
        # manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
        # pitch_tier = call(manipulation, "Extract pitch tier")
        #
        # call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)
        #
        # call([pitch_tier, manipulation], "Replace pitch tier")
        # sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")
        # sound_octave_up.save(f'./dist/male/{file}.wav', "WAV")
        # print(label)
        y, sr = librosa.load(filename, sr=None)
        y_slow = librosa.effects.time_stretch(y, rate=0.9)
        sf.write(f'./dist/slowwav/{file}.wav', y_slow, sr)
        # y = librosa.effects.pitch_shift(y, sr, n_steps=-3)
        # sf.write(f'./dist/male/{file}.wav', sound_octave_up.values, sound_octave_up.sampling_frequency)
    for index in range(len(label_df1)):
        label = label_df1.loc[index, 'label']
        filename = os.path.join(wav_path1, label_df1.loc[index, 'filename'])
        file = label_df1.loc[index, 'filename'].split('\\')[-1].split('.')[0]
        print(filename)
        y, sr = librosa.load(filename, sr=None)
        sf.write(f'./dist/slowwav/{file}.wav', y, sr)
    for index in range(len(label_df2)):
        label = label_df2.loc[index, 'label']
        filename = os.path.join(wav_path1, label_df2.loc[index, 'filename'])
        file = label_df2.loc[index, 'filename'].split('\\')[-1].split('.')[0]
        print(filename)
        y, sr = librosa.load(filename, sr=None)
        sf.write(f'./dist/slowwav/{file}.wav', y, sr)
    #     # print(label)
    #     y, sr = librosa.load(filename, sr=None)
    #     sf.write(f'./dist/adjustwav/{file}.wav', y, sr)










        # i=0
    # y,sr=librosa.load('./dist/alterwav/devel_0001.wav',sr=None)
    # # print(sr)
    # for filename in tqdm(wavs):
    #     file=filename.split('\\')[-1].split('.')[0]
    #     print(file)
    #     y,sr=librosa.load(filename,sr=None)
    #     # y_slow = librosa.effects.time_stretch(y, rate=0.9)
    #     sf.write(f'./dist/highwav/{file}.wav', y*1.1, sr)
        # n_fft=1024
        # mag=np.abs(librosa.core.stft(y,n_fft=n_fft,hop_length=10,win_length=441,window='hamming'))
        # mag_log=20*np.log(mag)
        # D=librosa.amplitude_to_db(mag,ref=np.max)
        # pylab.axis('off')  # no axis
        # pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        # librosa.display.specshow(D,sr=sr,hop_length=10,x_axis='s',y_axis='linear')
        # file=filename.split('\\')[-1].split('.')[0]
        # pylab.savefig(f'./dist/features/spectrum/{file}.jpg', bbox_inches=None, pad_inches=0)
        # i+=1
        # pylab.close()
    # jpgs=glob.glob(f'./dist/features/spectrum/example_*.jpg')
    # for jpg in jpgs:
    #     data=Image.open(jpg)
    #     data=np.array(data)
    #     print(data.shape)










































    #
    # get_word_list("涉江采芙\蓉ab,.兰泽多。芳草")
