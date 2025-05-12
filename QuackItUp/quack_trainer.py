import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from cnn import CNNNetwork
from audio_dataset import DuckDataset
import torchaudio

# 1) Download Dataset
# 2) Create data loader
# 3) Build Model
# 4) Train
# 5) Save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001



def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backpropagation
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
        
# 1) Download Dataset
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ANNOTATIONS_FILE = "DuckDataset\\audio_files.csv"
AUDIO_DIR = "DuckDataset\\Clean"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
    )

dd = DuckDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
print("Accessed Dataset")

# 2) Create data loader
train_data_loader = create_data_loader(dd, batch_size=BATCH_SIZE)

# 3) Build Model
print(f"Using {device} device")
cnn = CNNNetwork()

# 4) Train
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(),lr = LEARNING_RATE)
train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

# 5) Save trained model
torch.save(cnn.state_dict(), "feedforwardnet.pth")
print("Model trained and stored at feedforwardnet.pth")
    
    