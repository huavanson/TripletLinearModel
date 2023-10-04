import torch
import torch.nn as nn


class AirModel(nn.Module):
    def __init__(self, input_size):
        super(AirModel, self).__init__()
        # self.fc = nn.Linear(25, 1)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # out = self.fc(x)
        # out = self.relu(out)
        return x.squeeze(1)
    
class AirModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=25, hidden_size=1, num_layers=1)
        self.linear = nn.Linear(1, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.relu(x)
        return x.squeeze(1)