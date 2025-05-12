import torch.nn.functional as F
import torchaudio
import torch

def preprocess_audio(path, transform, target_sample_rate, num_samples, device="cpu"):
    signal, sr = torchaudio.load(path)
    signal = signal.to(device)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate).to(device)
        signal = resampler(signal)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    elif signal.shape[1] < num_samples:
        signal = F.pad(signal, (0, num_samples - signal.shape[1]))

    mel = transform(signal)
    mel = mel.unsqueeze(0)  # Add batch dimension
    return mel