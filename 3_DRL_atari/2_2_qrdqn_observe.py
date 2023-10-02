##### PACKAGES

from torch import nn
import torch
import torch.nn.functional as F
import gym
from collections import deque
import itertools
import numpy as np
import random
import math # math.isnan
import os
import time
import psutil
from torch.utils.tensorboard import SummaryWriter # needed for displaying in tensorboard (need to do pip install tensorboard. needs to be above specific version.)

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor # stable_baselines3 requires gym==0.17.3
from wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as mspack_numpy_patch
mspack_numpy_patch() # needed to save the trained model.


##### Tuning Parameters

### Simple Setting
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 1000
LR = 2.5e-4
min_sq_grad = None # epsilon_adam
SAVE_PATH = './QRDQN_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/QRDQN_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# use_cuda = False
# dummy_or_subproc = "dummy"
dummy_or_subproc = "subproc"
seed=1
# seed=None
n_quant = 200
kappa=1.0
# kappa=0.01 # should be 0, but torch does not support 0.


##### Neural Network

quants = np.linspace(0.0, 1.0, n_quant + 1)[1:]
quants_target = (np.linspace(0.0, 1.0, n_quant + 1)[:-1] + quants)/2 # making the quantiles to be the midpoints of the bins.
huberloss=torch.nn.HuberLoss(reduction='none', delta=kappa) # reduction='none' : elemetwise (all pairs of x-y). 'sum' and 'mean' returns one value only. => same for F.smooth_l1_loss.

def nature_cnn(observation_space, depths=(32,64,64), final_layer=512): 
    n_input_channels = observation_space.shape[0] # observation_space.shape: (4, 84, 84) in breakout.
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4), # (mb, 4, 84, 84) -> (mb, 32, 20, 20)
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2), # (mb, 32, 20, 20) -> (mb, 64, 9, 9)
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1), # (mb, 64, 9, 9) -> (mb, 64, 7, 7)
        nn.ReLU(),
        nn.Flatten() # (mb, 64*7*7)
    )

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1] # no need to convert to cuda.
    
    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU()) # (mb, 64*7*7) -> (mb, 512)

    return out


class Network(nn.Module): 
    # Network instance (e.g. online_net) has attribute .device, but nn.Sequential (e.g.conv_net: nature_cnn instance) does not have it.
    # But interestingly, we can also do conv_net.to(device) and it will works with conv_net(x) if conv_net.device == x.device.

    def __init__(self, env, device, n_quant): # every parameter that will be a part of the NN model should be within init. (but the activation function may be in forward.)
        super().__init__()
        self.num_actions = env.action_space.n
        self.device = device 
        conv_net = nature_cnn(env.observation_space)
        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions * n_quant)) # add self.n_atom


    def forward(self, x): # x: (mb_size, 84, 84, 4) tensor 
        # There are some caution when checking this out.
        # Caution1: Only torch-tensors are allowed, not numpy array.
        # Caution2: The input tensor should be in the same device as the model. (may need to convert to cuda.)
        mb_size = x.size(0) # mini-batch size
        action_values = self.net(x).view(mb_size, self.num_actions, n_quant) # (mb_size, num_actions, n_quant)
        return action_values

    def act(self, obses, epsilon): # mb_size = num_envs in act, but mb_size = BATCH_SIZE in compute_loss.
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device) # Everywhere there is torch.as_tensor, we need to put "device=self.device".
        q_distribution = self(obses_t) # self(obses_t) returns 2d-array of (mb_size, (num_actions x n_quant)), whereas self(obses_t) returns 3d-array of (mb_size, num_actions, n_quant).
        q_values = q_distribution.mean(dim=2) # (mb_size, num_actions)
        actions = torch.argmax(q_values, dim=1).data.cpu().tolist() # taking maximum along dim=1 (actions). returns (mb_size, )

        for i in range(len(actions)):            
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    
    def compute_loss(self, transitions, target_net):

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray([obses])
            new_obses = np.asarray([new_obses])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # needs to be a column vector
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)     # needs to be a column vector
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)   # needs to be a column vector
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)


        ## Compute evaluation distribution
        eval_q_distribution = self(obses_t) # mb_size x num_actions x n_quant
        mb_size = eval_q_distribution.size(0)
        eval_q_dist = torch.stack([eval_q_distribution[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # Z_online(s_t,a_t) : fixed action => mb_size x n_quant

        ## Compute next state distribution
        # next_q_distribution = target_net(new_obses_t) # mb_size x num_actions x n_quant
        next_q_distribution = target_net(new_obses_t).detach() # may detach: Target-related tensors do not take part in the gradient calculation. => So .detach() is used in .compute_loss(, not act.()).
        next_q_values = next_q_distribution.mean(dim=2) # mb_size x num_actions
        best_actions = torch.argmax(next_q_values, dim=1) # mb_size
        next_q_dist = torch.stack([next_q_distribution[i][best_actions[i]] for i in range(mb_size)]) # mb_size x n_quant
        target_q_dist = rews_t + GAMMA * (1-dones_t) * next_q_dist

        # next_q_dist = next_q_dist.data.cpu().numpy() # .data : remove gradient & .cpu() : remove gpu & .numpy() : convert to numpy array.        
        # target_q_dist = np.expand_dims(rews, 1) + GAMMA * np.expand_dims((1. - dones),1) * next_q_dist # mb_size x n_quant (broadcasting)

        ## Compute loss => Objective: Huber loss (Dabney)
        eval_q_dist = eval_q_dist.unsqueeze(2) # mb_size x n_quant x 1 # unsqueeze in torch = expand_dims in numpy
        target_q_dist = target_q_dist.unsqueeze(1) # mb_size x 1 x n_quant
        # u_values = target_q_dist - eval_q_dist # mb_size x n_quant x n_quant
        u_values = target_q_dist.detach() - eval_q_dist # may detach
        tau_values = torch.as_tensor(quants_target, dtype=torch.float32, device=self.device).view(1,-1,1) # 1 x n_quant x 1        
        weight = torch.abs(tau_values - u_values.le(0).float()) # mb_size x n_quant x n_quant # Logical values should be switched into float. 

        # rho_values = huberloss(eval_q_dist, target_q_dist) # mb_size x n_quant x n_quant (sample by eval by target)
        rho_values = huberloss(eval_q_dist, target_q_dist.detach()) / kappa # may detach
        loss_bybatch = (weight*rho_values).mean(dim=2).sum(dim=1) # mean over target, sum over eval. => mb_size 
        loss = loss_bybatch.mean() # mean over samples

        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        
        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}

        self.load_state_dict(params)


        

device = torch.device('cpu')

input_envs = [lambda: make_atari_deepmind('BreakoutNoFrameskip-v4', seed=i, scale_values=True) for i in range(1)] # lambda: has to stay in a functional form. (same with dqn8.py)
vec_env = DummyVecEnv(input_envs) 
env = BatchedPytorchFrameStack(vec_env, k=4) # contains converting to lazy frames. What's more, it applies to VecEnv.

net = Network(env, device, n_quant=n_quant)
net = net.to(device)

net.load('./models/2_qrdqn/4th_copy/f1.pack') # QR-DQN-1. seed=1
# net.load('./models/2_qrdqn/4th_copy/f2.pack') # QR-DQN-1. seed=2
# net.load('./models/2_qrdqn/4th_copy/f3.pack') # QR-DQN-1. seed=3
# net.load('./models/2_qrdqn/4th_copy/g1.pack') # QR-DQN-0. seed=4
# net.load('./models/2_qrdqn/4th_copy/g2.pack') # QR-DQN-0. seed=5
# net.load('./models/2_qrdqn/4th_copy/g3.pack') # QR-DQN-0. seed=6 # terrible for some reason.



obses=env.reset()
new_stage = True
prev_life = 5
for t in itertools.count():

    a=env.render()
    # time.sleep(0.02)

    if isinstance(obses[0], PytorchLazyFrames):
        act_obses = np.stack([o.get_frames() for o in obses])
        actions = net.act(act_obses, 0.0)
    else:
        actions = net.act(obses, 0.0)

    # action = net.act(obs, 0.0)
    if new_stage:
        actions=[1] # If action=2 in the new_stage, the game suddenly stops for some reason.

    obses, rew, done, info = env.step(actions)
    life = info[0]['ale.lives']
    if life < prev_life:
        new_stage=True
    else:
        new_stage=False
    
    prev_life = life

    if done[0]:
        obses=env.reset()


