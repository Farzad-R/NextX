import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaLSTM(nn.Module):
    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=18, hidden_size=168,
                            batch_first=True, num_layers=1, bidirectional=False)
        self.fc2 = nn.Linear(168, 12)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc2(out)
        return out

class LSTMDENSE(nn.Module):
    def __init__(self):
        super(LSTMDENSE, self).__init__()
        self.lstm = nn.LSTM(input_size=18, hidden_size=168,
                            batch_first=True, num_layers=1, bidirectional=False)

        self.fc1 = nn.Linear(168, 24)
        self.fc2 = nn.Linear(24, 12)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out