import torch.nn as nn

class FF(nn.Module):
    def __init__(self, in_channels: int, in_dim: tuple[int, int], 
                 width: int, out_features: int):
        super().__init__()
        self.in_features = in_channels * in_dim[0] * in_dim[1]
        self.width = width
        self.out_features = out_features
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.in_features, width)
        self.fc2 = nn.Linear(width, out_features)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class CNN(nn.Module):
    def __init__(self, in_channels, in_dim: tuple[int, int], out_features):
        super(CNN, self).__init__()
        self.in_dim = in_dim
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 7 * 7, out_features)  # assumes input 28x28

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
