import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class UrbanSoundDataset(Dataset):
    def __init__(obj, annotations_file, audio_dir, transformation, target_sample_rate):
        obj.annotations = pd.read_csv(annotations_file)
        obj.audio_dir = audio_dir
        obj.transformation = transformation
        obj.target_sample_rate = target_sample_rate

    
    def __len__(obj):
        return len(obj.annotations)

    def __getitem__(obj, index):
        audio_sample_path = obj.__get_audio_sample_path(index)
        label = obj.__get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = obj._resample_if_necessary(signal, sr)
        signal = obj._mix_down_if_necessary(signal) 
        signal = obj.transformation(signal)

        return signal,label
    
    def _resample_if_necessary(obj, signal, sr):
        if sr != obj.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, obj.target_sample_rate)
            signal = resampler(signal)

        return signal
    
    def _mix_down_if_necessary(obj, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal
    
    def __get_audio_sample_path(obj,index):
        fold = f"fold{obj.annotations.iloc[index, 5]}"

        path = os.path.join(obj.audio_dir, fold, obj.annotations.iloc[index, 0])
        return path
    
    def __get_audio_sample_label(obj, index):
        return obj.annotations.iloc[index, 6]
    


### Run ###############################################################################################
ANNOTATIONS_FILE = "C:/Users/shawn/OneDrive/Documents/CodeResources/SoundClassifier/UrbanSoundDataset/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "C:/Users/shawn/OneDrive/Documents/CodeResources/SoundClassifier/UrbanSoundDataset/UrbanSound8K/UrbanSound8K/audio"
SAMPLE_RATE = 16000

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
    )

usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)

print(f"There are {len(usd)} samples in the dataset")

signal, label = usd[0]

a=1
