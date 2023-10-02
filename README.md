# 1. Requirements
- Python 3.9.13 (or higher)
- torch 1.13.1+cu116
- Ingrain Roms.py into python. This is feasible by running "python -m atari_py.import_roms".

# 2. Code Explanation

## 1. review_torch
- contains basic usage of deep neural networks based on pytorch.
- This includes 1) regression, 2) logistic regression, 3) sine-curve matching. 

## 2. DQL
- contains well-known examples of basic Deep Q Learning, starting from 0) template of neural network.
- This includes 1) cartpole, 2) mountain car examples.

## 3 DRL_atari
- contains wrappers that contain necessary functions.
- Every method has (method).py and (method_observe).py, each of which is for training the model and demostrating the play of the model. 
0) (Mnih et al., 2015) Human-level control through deep reinforcement learning
1) (Bellemare et al., 2017a) Distributional perspective on reinforcement learning
2) (Dabney et al., 2018b) Distributional reinforcement learning with quantile regression
3) (Dabney et al., 2018a) Implicit quantile networks for distributional reinforcement learning
4) (Nguyen et al., 2021) Distributional reinforcement learning via moment matching

