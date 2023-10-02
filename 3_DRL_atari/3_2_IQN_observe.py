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
SAVE_PATH = './IQN_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/IQN_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# use_cuda = False
# dummy_or_subproc = "dummy"
dummy_or_subproc = "subproc"
seed=1
# seed=None
kappa=1.0
# kappa=0.01 # should be 0, but torch does not support 0.



##### Neural Network

huberloss=torch.nn.HuberLoss(reduction='none', delta=kappa) 

def nature_cnn(observation_space, depths=(32,64,64)): 
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
    

    return cnn, n_flatten

    # out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU()) # (mb, 512) # ==> 여기선 취소하자.



class Network(nn.Module): 
    # Network instance (e.g. online_net) has attribute .device, but nn.Sequential (e.g.conv_net: nature_cnn instance) does not have it.
    # But interestingly, we can also do conv_net.to(device) and it will works with conv_net(x) if conv_net.device == x.device.

    def __init__(self, env, device, n_quant, final_layer=512): # every parameter that will be a part of the NN model should be within init. (but the activation function may be in forward.)
        super().__init__()
        self.num_actions = env.action_space.n
        self.device = device 

        conv_net, middle_dim = nature_cnn(env.observation_space) # until flatten (mb, 64*7*7)

        ## psi (CNN): image (mb, 4, 84, 84) -> (mb, 64*7*7)
        self.conv_net = conv_net

        ## phi: cos-tranformed (mb, 64) -> (mb, 64*7*7)
        self.middle_dim = middle_dim        
        self.phi = nn.Sequential(nn.Linear(n_cos_trans, self.middle_dim), nn.ReLU()) # (mb, n_cos_trans) -> (mb, 64*7*7)

        ## final: (mb, 64*7*7) -> (mb, num_actions*n_quant)
        self.finals = nn.Sequential(nn.Linear(self.middle_dim, final_layer), nn.ReLU(), nn.Linear(final_layer, self.num_actions)) 

    def forward(self, x):
        # mb_size = x.size(0)
        psi_x = self.conv_net(x) # (mb, 64*7*7)
        tau = torch.rand(n_quant, 1).to(device) # (n_quant, 1)
        quants = torch.arange(0, n_cos_trans, 1.0).to(device).unsqueeze(0) # (1, n_cos_trans)
        cos_trans = torch.cos(tau * quants * 3.141592) # (n_quant, n_cos_trans)
        phi_tau = self.phi(cos_trans) # (n_quant, 64*7*7)

        psi_x = psi_x.unsqueeze(1) # (mb, 1, 64*7*7) # reorganize for broadcasting.
        phi_tau = phi_tau.unsqueeze(0) # (1, n_quant, 64*7*7)
        interaction = psi_x * phi_tau # (mb, n_quant, 64*7*7) 

        F_values = self.finals(interaction) # (mb, n_quant, num_actions)
        action_values = F_values.transpose(1, 2) # (m, num_actions, n_quant)

        return action_values, tau # (m, num_actions, n_quant), (n_quant, 1)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device) # Everywhere there is torch.as_tensor, we need to put "device=self.device".
        action_values, tau = self(obses_t)
        q_values = action_values.mean(dim=2)
        actions = torch.argmax(q_values, dim=1).data.cpu().tolist() 

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
        eval_q_distribution, tau_eval = self(obses_t) # (mb_size, num_actions, n_quant), (n_quant, 1)
        mb_size = eval_q_distribution.size(0)
        eval_q_dist = torch.stack([eval_q_distribution[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # Z_online(s_t,a_t) : fixed action => (mb_size, n_quant)

        ## Compute next state distribution
        next_q_distribution, tau_target = target_net(new_obses_t)
        next_q_distribution = next_q_distribution.detach() # (mb_size, num_actions, n_quant)

        next_q_values = next_q_distribution.mean(dim=2) # (mb_size, num_actions)
        best_actions = torch.argmax(next_q_values, dim=1) # (mb_size,)
        next_q_dist = torch.stack([next_q_distribution[i][best_actions[i]] for i in range(mb_size)]) # (mb_size, n_quant)
        target_q_dist = rews_t + GAMMA * (1-dones_t) * next_q_dist # (mb_size, n_quant)


        ## Compute loss => Objective: Huber loss (Dabney)
        eval_q_dist = eval_q_dist.unsqueeze(2) # (mb_size, n_quant: eval, 1) # unsqueeze in torch = expand_dims in numpy
        target_q_dist = target_q_dist.unsqueeze(1) # (mb_size, 1, n_quant: target)
        u_values = target_q_dist.detach() - eval_q_dist # may detach # (mb_size, n_quant: eval, n_quant: target)
        tau_values = torch.as_tensor(tau_eval, dtype=torch.float32, device=self.device).view(1,-1,1) # (1, n_quant : eval, 1)        
        weight = torch.abs(tau_values - u_values.le(0).float()) # (mb_size, n_quant: eval, n_quant: target)
    
        rho_values = huberloss(eval_q_dist, target_q_dist.detach()) / kappa # (mb_size, n_quant: eval, n_quant: target) => may detach
        loss_bybatch = (weight*rho_values).mean(dim=2).sum(dim=1) # (mb_size,) mean over target, sum over eval. => mb_size 
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

# n_quant = 64; modelindex=1 # If we don't match n_quant (and thereby the network structure), it still works fine, but gives us an error message.
# n_quant = 64; modelindex=2
# n_quant = 64; modelindex=3
# n_quant = 32; modelindex=4
# n_quant = 32; modelindex=5
# n_quant = 32; modelindex=6
# n_quant = 8; modelindex=7
# n_quant = 8; modelindex=8
n_quant = 8; modelindex=9

n_cos_trans = n_quant
net = Network(env, device, n_quant=n_quant)
net = net.to(device)

net.load('models/3_IQN/5th_copy/model' + str(modelindex) + '.pack') 
# CAUTION!!! Make sure we put the address correctly.
# Even if we don't, the code runs fine, but gives us terrible results. This is because the network stays the same as before we loaded the saved model.
# This is the scariest thing that can happen. It is erroneous, but it works in the for loop below.

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


