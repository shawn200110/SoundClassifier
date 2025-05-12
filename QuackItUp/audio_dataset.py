import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import numpy as np



class DuckDataset(Dataset):
    def __init__(obj,annotations_file,audio_dir,transformation,target_sample_rate,num_samples,device):
        obj.annotations = pd.read_csv(annotations_file)
        obj.audio_dir = audio_dir
        obj.device = device
        obj.transformation = transformation.to(obj.device)
        obj.target_sample_rate = target_sample_rate
        obj.num_samples = num_samples

    
    def __len__(obj):
        return len(obj.annotations)

    def __getitem__(obj, index):
        audio_sample_path = obj._get_audio_sample_path(index)
        label = obj._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(obj.device)
        signal = obj._resample_if_necessary(signal, sr)
        signal = obj._mix_down_if_necessary(signal) 
        signal = obj._truncate_or_pad(signal)
        signal = obj.transformation(signal)


        return signal,label
    
    def _truncate_or_pad(obj,signal):
        if signal.shape[1] > obj.num_samples:
            signal = signal[:,:obj.num_samples]
        elif signal.shape[1] < obj.num_samples:
            # right pad
            num_zeros = obj.num_samples - signal.shape[1]
            last_dim_padding = (0,num_zeros)
            signal = torch.nn.functional.pad(signal,last_dim_padding)

        return signal

            
    
    def _resample_if_necessary(obj, signal, sr):
        if sr != obj.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, obj.target_sample_rate)
            signal = resampler(signal)

        return signal
    
    def _mix_down_if_necessary(obj, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal
    
    def _get_audio_sample_path(obj, index):
        if index >= 50:
            subfolder = "NotDuck"
        else:
            subfolder = "Duck"
            
        filename = obj.annotations.iloc[index, 0]
        path = os.path.join("DuckDataset", "Clean", subfolder, filename)
        return path
        
    def _get_audio_sample_label(obj, index):
        label_str = obj.annotations.iloc[index, 2]
        label = 0 if label_str == 'duck_quack' else 1
        return torch.tensor(label)
    
    


### Run ###############################################################################################
ANNOTATIONS_FILE = "DuckDataset\\audio_files.csv"
AUDIO_DIR = "DuckDataset\\Clean"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        )

    usd = DuckDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[1]


    # indexes = np.array([0,5,10,15,20,25,55,60,65,70,75,80])


    # for i in range(len(indexes)):  
        
    #     signal, label = usd[indexes[i]]
    #     mel = signal.squeeze().cpu().numpy()
    #     plt.subplot(4, 3, i+1) 
    #     plt.imshow(mel, origin='lower', aspect='auto', cmap='magma')
    #     plt.title(indexes[i])
    #     plt.tight_layout()
        
    # plt.show()