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


### Bellemare's Setting: LR changed, eps specified
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.1
EPSILON_DECAY=int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 2.5e-4
min_sq_grad = 0.01 / BATCH_SIZE # eps
SAVE_PATH = './C51_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/C51_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# dummy_or_subproc = "dummy" # may be safer to run this in school cluster.
dummy_or_subproc = "subproc"
seed=1
# seed=None
n_atom = 51
V_min = -10.
V_max = 10.
V_range = np.linspace(V_min, V_max, n_atom)
V_step = ((V_max-V_min)/(n_atom-1))



##### Neural Network

def nature_cnn(observation_space, depths=(32,64,64), final_layer=512):
    n_input_channels = observation_space.shape[0]
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1] # no need to convert to cuda.
    
    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):

    def __init__(self, env, device, n_atom): # every parameter that will be a part of the NN model should be within init. (but the activation function may be in forward.)
        super().__init__()
        self.num_actions = env.action_space.n
        self.device = device 
        self.n_atom = n_atom
        self.V_range_cpu = torch.FloatTensor(V_range) # always cpu => for numpy
        self.V_range = torch.as_tensor(V_range, dtype=torch.float32, device=self.device)

        conv_net = nature_cnn(env.observation_space)
        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions * self.n_atom)) # add self.n_atom

    def forward(self, x): # target_net(obses_t) implemets this. (obses_t : batch of obs) => x must be a torch.tensor, not np.array.
        mb_size = x.size(0) # mini-batch size
        q_distribution = F.softmax(self.net(x).view(mb_size, self.num_actions, self.n_atom), dim=2) # mb_size x num_actions x n_atom, but each representing probability.
        return q_distribution

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device) # Everywhere there is torch.as_tensor, we need to put "device=self.device".
        q_distribution = self(obses_t)
        q_values = torch.sum(q_distribution * self.V_range.view(1, 1, -1), dim=2)
        actions = torch.argmax(q_values, dim=1).tolist()

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
        # rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)     # needs to be a column vector
        # dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)   # needs to be a column vector
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        ## Compute evaluation distribution
        eval_q_distribution = self(obses_t) # mb_size x num_actions x n_atom
        mb_size = eval_q_distribution.size(0)
        eval_q_dist = torch.stack([eval_q_distribution[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # Q_online(s_t,a_t) : fixed action

        ## Compute next distribution
        next_q_distribution = target_net(new_obses_t) # mb_size x num_actions x n_atom
        next_q_values = torch.sum(next_q_distribution * self.V_range.view(1, 1, -1), dim=2) # mb_size x num_actions
        best_actions = next_q_values.argmax(dim=1) 
        next_q_dist = torch.stack([next_q_distribution[i][best_actions[i]] for i in range(mb_size)]) # mb_size x n_atom
        next_q_dist = next_q_dist.data.cpu().numpy() # .data : remove gradient & .cpu() : remove gpu & .numpy() : convert to numpy array.
        # next_q_dist.sum(axis=1) # does not exactly become 1.0.

        ## Categorical projection
        next_v_range = np.expand_dims(rews, 1) + GAMMA * np.expand_dims((1. - dones),1) * np.expand_dims(self.V_range_cpu,0)
        next_v_pos = np.zeros_like(next_v_range)
        next_v_range = np.clip(next_v_range, V_min, V_max)
        next_v_pos = (next_v_range - V_min) / V_step

        target_q_dist = np.zeros((mb_size, n_atom)) # mb_size x n_atom
        lb = np.floor(next_v_pos).astype(int)
        ub = np.minimum(lb+1, 50)
        lb[ub==50] = 49
        # lb; ub
        # np.sum(ub-lb!=1)

        # begin=time.time()
        left_update = (next_q_dist * (ub - next_v_pos))
        right_update = (next_q_dist * (next_v_pos - lb))
        for i in range(mb_size):
            for j in range(n_atom):
                target_q_dist[i, lb[i,j]] += left_update[i,j]
                target_q_dist[i, ub[i,j]] += right_update[i,j]
        target_q_dist = torch.as_tensor(target_q_dist, dtype=torch.float32, device=self.device)
        # target_q_dist.sum(axis=1) # does not exactly become 1.0.
        # end = time.time()
        # end-begin

        ## Compute loss
        loss = target_q_dist * ( - torch.log(eval_q_dist + 1e-8)) # cross entropy
        loss = loss.sum(dim=1).mean() # mb_size x n_atom -> mb_size -> scalar
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


net = Network(env, device, n_atom=n_atom)
net = net.to(device)

# net.load('./models/1_c51/3rd_copy/seed1.pack')
# net.load('./models/1_c51/3rd_copy/seed2.pack')
# net.load('./models/1_c51/3rd_copy/seed3.pack') 
# net.load('./models/1_c51/3rd_copy/seed4.pack') 
net.load('./models/1_c51/3rd_copy/seed5.pack') 



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


