import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1) Download Dataset
# 2) Create data loader
# 3) Build Model
# 4) Train
# 5) Save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

class FeedForwardNet(nn.Module):

    def __init__(obj):
        super().__init__()

        obj.flatten = nn.Flatten()
        obj.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

        obj.softmax = nn.Softmax(dim=1)

    def forward(obj, input_data):
        flattened_data = obj.flatten(input_data)
        logits = obj.dense_layers(flattened_data)
        predictions = obj.softmax(logits)
        return predictions



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
        
# 1) Download Dataset
train_data,_ = download_mnist_datasets()
print("MNIST dataset downloaded")

# 2) Create data loader
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

# 3) Build Model
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")
feed_forward_net = FeedForwardNet().to(device)

# 4) Train
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(feed_forward_net.parameters(),lr = LEARNING_RATE)
train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)
torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
print("Model trained and stored at feedforwardnet.pth")


# 5) Save trained model
    
    