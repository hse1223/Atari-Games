import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self): # Once we run (or make) model=Model(), it generates the below.
        super().__init__() # necessary part
        self.layer1 = nn.Linear(128, 32) # size=128 input -> size=32 output
        self.layer2 = nn.Linear(32, 16)  # size=32 input -> size=16 output
        self.layer3 = nn.Linear(16, 1)   # size=16 input -> size=1 output
    def forward(self, features): # features is the input of the instance. => model(features) gives us the predicts.
        x = self.layer1(features) # So this is what happens when we actually run model(features).
        x = F.relu(x)  
        x = self.layer2(x)
        x = F.relu(x)  
        x = self.layer3(x)
        return x

model = Model()
features = torch.rand((2,128)) # input: 2 samples of size=128. shape=(2, 128)
model(features)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, features) :
        x = self.base(features)
        return x

model = Model()
features = torch.rand((2,128)) # input: 2 samples of size=128. shape=(2, 128)
model(features)


