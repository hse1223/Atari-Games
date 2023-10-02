# https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be 

##### STEP1: Generate data

import numpy as np 
N = 1000; p = 3

np.random.seed(0)
Xdat = np.random.normal(size=N*p).reshape(N,p)
beta = np.linspace(start=-20., stop=20, num=p+1) / 10
beta = np.round(beta)

ones = np.array([1.]*N)
Xmat = np.hstack((ones.reshape(-1,1), Xdat))
Xb = np.matmul(Xmat, beta)

import torch
prob_vec = torch.sigmoid(torch.from_numpy(Xb))
Yvec = np.random.binomial(1, prob_vec, N)


# ##### sklearn package

# from sklearn import linear_model
# # logr = linear_model.LogisticRegression(penalty='none')
# logr = linear_model.LogisticRegression()
# logr.fit(Xdat, Yvec)
# np.concatenate([logr.intercept_, logr.coef_.reshape(p)]) # estimated
# beta # true


##### STEP2: transform it into torch tensors.

torch.set_default_dtype(torch.float64)
Xdat_torch = torch.from_numpy(Xdat.astype(np.float64))
Y_torch = torch.from_numpy(Yvec.astype(np.float64)) # convert to float64
Y_torch = Y_torch.reshape(N,1)


##### STEP3: dataloader – ready for shuffling.

from torch.utils.data import TensorDataset
train_ds = TensorDataset(Xdat_torch, Y_torch)
from torch.utils.data import DataLoader
batch_size = int(N/10)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


##### STEP4: Choose model.

from torch import nn
class LogisticRegression(nn.Module):
     def __init__(self, input_dim):
        # super(LogisticRegression, self).__init__() # option 1
        super().__init__() # option 2
        self.linear = torch.nn.Linear(input_dim, 1)
     def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

model = LogisticRegression(p)
torch.manual_seed(0)
num_epochs=1000
criterion = nn.BCELoss()

##### STEP5: Backpropagation

opt = torch.optim.SGD(model.parameters(), lr=1e-2)
for epoch in range(num_epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

list(model.parameters()) # how we turn a generator into a list.
beta




