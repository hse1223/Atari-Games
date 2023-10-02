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
# LR = 2.5e-4
LR = 5e-5
min_sq_grad = None # epsilon_adam
# n_particle = 30
# bandwidth = [5] # single bandwidth
# bandwidth = [10] # single bandwidth
bandwidth=[1,2,3,4,5,6,7,8,9,10] # 1~10 (mixture)
SAVE_PATH = './MMDQN_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/MMDQN_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# use_cuda = False
dummy_or_subproc = "dummy"
# dummy_or_subproc = "subproc"
# seed=1
# seed=None


##### Neural Network

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

    def __init__(self, env, device): # every parameter that will be a part of the NN model should be within init. (but the activation function may be in forward.)
        super().__init__()
        self.num_actions = env.action_space.n
        self.device = device 
        conv_net = nature_cnn(env.observation_space)
        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions * n_particle)) # add self.n_atom


    def forward(self, x): # x: (mb_size, 84, 84, 4) tensor 
        # There are some caution when checking this out.
        # Caution1: Only torch-tensors are allowed, not numpy array.
        # Caution2: The input tensor should be in the same device as the model. (may need to convert to cuda.)
        mb_size = x.size(0) # mini-batch size
        action_values = self.net(x).view(mb_size, self.num_actions, n_particle) # (mb_size, num_actions, n_particle)
        return action_values

    def act(self, obses, epsilon): # mb_size = num_envs in act, but mb_size = BATCH_SIZE in compute_loss.
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device) 
        q_distribution = self(obses_t) # (mb_size, num_actions, n_particles).
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
        eval_q_distribution = self(obses_t) # mb_size x num_actions x n_particle
        mb_size = eval_q_distribution.size(0)
        eval_q_dist = torch.stack([eval_q_distribution[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # Z_online(s_t,a_t) : fixed action => mb_size x n_particle

        ## Compute next state distribution
        next_q_distribution = target_net(new_obses_t).detach() # may detach: Target-related tensors do not take part in the gradient calculation. 
        next_q_values = next_q_distribution.mean(dim=2) # mb_size x num_actions
        best_actions = torch.argmax(next_q_values, dim=1) # mb_size
        next_q_dist = torch.stack([next_q_distribution[i][best_actions[i]] for i in range(mb_size)]) # mb_size x n_particle
        target_q_dist = rews_t + GAMMA * (1-dones_t) * next_q_dist

        ## Calculate (LHS/RHS - LHS/RHS) - method 2
        LHS = eval_q_dist.unsqueeze(2) # mb_size x n_particle x 1 
        RHS = target_q_dist.unsqueeze(2) # mb_size x n_particle x 1 
        LHS_subtr_LHS_sq = (LHS - LHS.transpose(1,2))**2
        LHS_subtr_RHS_sq = (LHS - RHS.transpose(1,2))**2
        RHS_subtr_RHS_sq = (RHS - RHS.transpose(1,2))**2


        ## Compute loss - assuming N=M, but efficient.
        LL_inside = LHS_subtr_LHS_sq.unsqueeze(3) / bandwidth
        LR_inside = LHS_subtr_RHS_sq.unsqueeze(3) / bandwidth
        RR_inside = RHS_subtr_RHS_sq.unsqueeze(3) / bandwidth
        k_LL = torch.exp(-LL_inside)
        k_LR = torch.exp(-LR_inside)
        k_RR = torch.exp(-RR_inside)
        k_sums = k_LL + k_RR - 2 * k_LR
        MMD_loss_bybatch = k_sums.sum(dim=3).mean(dim=(1,2)) # sum over different kernels, sum over N^2 (particle, particle)
        MMD_loss_bybatch[MMD_loss_bybatch < 0] = 0. # 이 부분이 다르긴 했다.
        MMD_loss = MMD_loss_bybatch.mean()
        
        return MMD_loss

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


# n_particle = 10; seed=1
# n_particle = 50; seed=1
n_particle = 100; seed=1 # Wow, it finished the game!
# n_particle = 200; seed=1
# n_particle = 200; seed=2

net = Network(env, device)
net = net.to(device)

net.load('models/4_MMDRL/gaussianmix_atom' + str(n_particle) + "_seed" + str(seed)+".pack") 


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


