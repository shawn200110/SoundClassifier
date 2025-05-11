import torch
from torch import nn
import torchaudio  
from torch.utils.data import DataLoader
from audio_dataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and updates weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("------------------")
    print("Training is done")



### Run ###############################################################################################
# Setup
ANNOTATIONS_FILE = "C:/Users/shawn/OneDrive/Documents/CodeResources/SoundClassifier/UrbanSoundDataset/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "C:/Users/shawn/OneDrive/Documents/CodeResources/SoundClassifier/UrbanSoundDataset/UrbanSound8K/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
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
# 1) Get Dataset

usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
print("MNIST dataset downloaded")

# 2) Create data loader
train_data_loader = create_data_loader(usd, batch_size=BATCH_SIZE)

# 3) Build Model

cnn = CNNNetwork().to(device)
print(cnn)

# 4) Train
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(),lr = LEARNING_RATE)
train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

# 5) Save trained model
torch.save(cnn.state_dict(), "cnn.pth")
print("Model trained and stored at cnn.pth")

    
    