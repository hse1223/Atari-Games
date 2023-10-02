N=10000
use_cuda = True
# use_cuda = False

import numpy as np
np.random.seed(0)
X = np.random.uniform(size=N, low=0.0, high=2*np.pi)
Y = np.sin(X)


import torch 
import torch.nn as nn 

if use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print('\n\n\nUsing device:', device, '\n\n\n')

class SineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)


X_torch = torch.from_numpy(X.astype(np.float32)).reshape(N,1) # The first dimension should be the number of samples. The rest are the dimensions of the input.
Y_torch = torch.from_numpy(Y.astype(np.float32)).reshape(N,1)
X_torch = X_torch.to(device)
Y_torch = Y_torch.to(device)

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(X_torch,Y_torch) 
batch_size = int(N/10) # If we use batch_size=N, then it is not a stochastic gradient descent.
# batch_size = N
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = SineModel()
model = model.to(device) # must assign a new object when switching to cuda.
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    # epoch=0
    for xb,yb in train_dl:
        # break
        pred = model(xb)
        # yb=yb.to(device)
        loss = nn.MSELoss()(pred,yb) # We need nn.MSELoss(), not nn.MSELoss, to construct a proper loss function.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print('epoch {}, loss {:.4f}'.format(epoch+1, loss.item()))

Y_pred = model(X_torch).detach().cpu().numpy().reshape(N) # get rid of the gradient & cuda.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Sometimes there is an issue when plotting torch-trained data. This line is to solve the issue.

import matplotlib.pyplot as plt
plt.scatter(X,Y_pred, label="pred")
plt.scatter(X,Y, label="true")
plt.axhline(y=0, color='black')
plt.legend(loc="upper left")
plt.show()


