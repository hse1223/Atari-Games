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
min_sq_grad = None # eps
SAVE_PATH = './C51_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/C51_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# use_cuda = False
# dummy_or_subproc = "dummy"
dummy_or_subproc = "subproc"
seed=1
# seed=None
n_atom = 51
V_min = -10.
V_max = 10.



# ### Previous DQN setting
# GAMMA=0.99
# BATCH_SIZE=32
# BUFFER_SIZE=int(1e6)
# MIN_REPLAY_SIZE=50000
# EPSILON_START=1.0
# EPSILON_END=0.1
# EPSILON_DECAY=int(1e6)
# NUM_ENVS = 4
# TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
# LR = 5e-5
# min_sq_grad = None # eps
# SAVE_PATH = './C51_model.pack'
# SAVE_INTERVAL = 10000
# LOG_DIR = './logs/C51_vanilla'
# LOG_INTERVAL = 1000
# use_cuda = True
# # dummy_or_subproc = "dummy" 
# dummy_or_subproc = "subproc"
# seed=1
# # seed=None
# n_atom = 51
# V_min = -10.
# V_max = 10.


# ### Bellemare's Setting: LR changed, eps specified
# GAMMA=0.99
# BATCH_SIZE=32
# BUFFER_SIZE=int(1e6)
# MIN_REPLAY_SIZE=50000
# EPSILON_START=1.0
# EPSILON_END=0.1 # This should be 0.01 (p.7 of Bellemare 2017). But 0.1 works decently well.
# EPSILON_DECAY=int(1e6)
# NUM_ENVS = 4
# TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
# LR = 2.5e-4
# min_sq_grad = 0.01 / BATCH_SIZE # eps
# SAVE_PATH = './C51_model.pack'
# SAVE_INTERVAL = 10000
# LOG_DIR = './logs/C51_vanilla'
# LOG_INTERVAL = 1000
# use_cuda = True
# # dummy_or_subproc = "dummy" 
# dummy_or_subproc = "subproc"
# seed=1
# # seed=None
# n_atom = 51
# V_min = -10.
# V_max = 10.



##### Neural Network

V_range = np.linspace(V_min, V_max, n_atom)
V_step = ((V_max-V_min)/(n_atom-1))

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
        self.n_atom = n_atom # In fact, we can just use n_atom instead of making an attribute self.n_atom. This is just a custom.
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

        ## Compute next state distribution
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


##### Set up the environment.

if use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if __name__ == "__main__": 

    input_envs = [lambda: make_atari_deepmind('BreakoutNoFrameskip-v4', seed=i, scale_values=True) for i in range(NUM_ENVS)] # lambda: has to stay in a functional form. (same with dqn8.py)

    if dummy_or_subproc=="dummy":
        vec_env = DummyVecEnv(input_envs) 
    elif dummy_or_subproc=="subproc":
        vec_env = SubprocVecEnv(input_envs) 
    else:
        raise ValueError("dummy_or_subproc must be either 'dummy' or 'subproc'")    

    env = BatchedPytorchFrameStack(vec_env, k=4) # contains converting to lazy frames. What's more, it applies to VecEnv.

    if seed!=None: # seed works properly, but differently between cpu and cuda.
        torch.manual_seed(seed)
        # env.seed(seed) # common seed. but we are going to use different seeds for each env.
        env.action_space.seed(seed) 
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed=seed)

    print('\n\n\n-------')
    print('device:', device)
    print(type(env.env))
    print('seed =', seed)
    print('-------\n\n\n')

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)
    episode_count = 0



    summary_writer = SummaryWriter(LOG_DIR)
    online_net = Network(env, device=device, n_atom=n_atom) 
    target_net = Network(env, device=device, n_atom=n_atom)
    online_net = online_net.to(device) 
    target_net = target_net.to(device)


    target_net.load_state_dict(online_net.state_dict())

    if min_sq_grad is None:
        optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)
    else:
        optimizer = torch.optim.Adam(online_net.parameters(), lr=LR, eps=min_sq_grad)


    ## Initialize replay buffer
    obses = env.reset()
    for _ in range(MIN_REPLAY_SIZE):

        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

        new_obses, rews, dones, _ = env.step(actions)

        for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

        obses = new_obses

    ## Main Training Loop
    start = time.time()
    obses = env.reset() 
    before=psutil.Process(os.getpid()).memory_info().rss/1024**2


    for step in itertools.count():

        # step=0 

        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                epinfos_buffer.append(info['episode'])
                episode_count += 1

        obses = new_obses

        ## Start Gradient Descent
        transitions = random.sample(replay_buffer, BATCH_SIZE) # 4 agents play in their own scenarios and collect their own data.
        loss = online_net.compute_loss(transitions, target_net) # But anyway, we are updating the model based on the resampled data.

        ## Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Net
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:

            if len(epinfos_buffer)==0:
                rew_mean = 0
                len_mean = 0
            else:
                rew_mean = np.mean([e['r'] for e in epinfos_buffer])    
                len_mean = np.mean([e['l'] for e in epinfos_buffer]) 
            
            print()
            print('Step:', step)
            print('Avg Rew:', rew_mean)
            print('Avg Ep Len', len_mean)
            print('Episodes:', episode_count)

            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('AvgEpLen', len_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)

            end = time.time()        
            print('Elapsed:', end-start, 'seconds')
            start = time.time()

            after=psutil.Process(os.getpid()).memory_info().rss/1024**2
            print('Current Memory: ', after)
            print('Memory Increment: ', after - before)
            before = after


        # Save Model
        if step % SAVE_INTERVAL == 0 and step!=0:
            print('Saving...')
            online_net.save(SAVE_PATH)    




# tensorboard --logdir ./logs # Run in cmd.



