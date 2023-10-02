##### Generate the data

import numpy as np

np.random.seed(0)
N=1000; p=3

mu = np.array([1.,3,5])
W = np.random.uniform(low=0, high=1, size=9).reshape(3,3)
Sigma = np.matmul(W, W.transpose())

Xdat = np.random.multivariate_normal(mean=mu, cov=Sigma, size=N)
ones = np.array([1.] * N)
Xmat = np.hstack((ones.reshape(N,1), Xdat))

beta = np.array([-5., 10, -5, 5])
error = np.random.normal(0, 1, N)
Yvec = np.matmul(Xmat, beta) + error


### STEP2: transform it into torch tensors.

import torch 
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
Xdat_torch = torch.from_numpy(Xdat)
Y_torch = torch.from_numpy(Yvec)
Y_torch = Y_torch.reshape(N,1)



### STEP3: dataloader – ready for shuffling.

from torch.utils.data import TensorDataset
train_ds = TensorDataset(Xdat_torch, Y_torch)
from torch.utils.data import DataLoader
batch_size = int(N/10)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


### STEP4: Choose model.

from torch import nn
torch.manual_seed(0)
model=nn.Linear(p,1)
num_epochs=10000


### STEP5: Backpropagation

import torch.nn.functional as F
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
for epoch in range(num_epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

list(model.parameters())