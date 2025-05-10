from torch.utils.data import Dataset
import os
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):
    def __init__(obj, annotations_file, audio_dir):
        obj.annotations = pd.read_csv(annotations_file)
        obj.audio_dir = audio_dir
    
    def __len__(obj):
        return len(obj.annotations)

    def __getitem__(obj, index):
        audio_sample_path = obj.__get_audio_sample_path(index)
        label = obj.__get_audio_sample_label(index)

        signal, sr = torchaudio.load(audio_sample_path)

        return signal,label
    
    def __get_audio_sample_path(obj,index):
        fold = f"fold{obj.annotations.iloc[index, 5]}"

        path = os.path.join(obj.audio_dir, fold, obj.annotations.iloc[index, 0])
        return path
    
    def __get_audio_sample_label(obj, index):
        return obj.annotations.iloc[index, 6]
    


### Run ###############################################################################################
ANNOTATIONS_FILE = ""
AUDIO_DIR = ""
usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

print(f"There are {len(usd)} samples in the dataset")

signal, label = usd[0]


