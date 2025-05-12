from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):

    def __init__(obj):
        super().__init__()
        # 4 convolution blocks / flatten / linear / softmax
        obj.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        obj.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        obj.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        obj.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        obj.flatten = nn.Flatten()
        obj.linear = nn.Linear(128*5*4, 10)
        obj.softmax = nn.Softmax(dim=1)

    def forward(obj,input_data):
        x = obj.conv1(input_data)
        x = obj.conv2(x)
        x = obj.conv3(x)
        x = obj.conv4(x)
        x = obj.flatten(x)
        logits = obj.linear(x)
        predictions = obj.softmax(logits)
        return predictions


cnn = CNNNetwork()
summary(cnn, (1,64,44))