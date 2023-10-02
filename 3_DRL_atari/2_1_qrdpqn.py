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

# ### Simple Setting
# GAMMA=0.99
# BATCH_SIZE=32
# BUFFER_SIZE=50000
# MIN_REPLAY_SIZE=1000
# EPSILON_START=1.0
# EPSILON_END=0.02
# EPSILON_DECAY=10000
# NUM_ENVS = 4
# TARGET_UPDATE_FREQ = 1000
# LR = 2.5e-4
# min_sq_grad = None # epsilon_adam
# SAVE_PATH = './QRDQN_model.pack'
# SAVE_INTERVAL = 10000
# LOG_DIR = './logs/QRDQN_vanilla'
# LOG_INTERVAL = 1000
# use_cuda = True
# # use_cuda = False
# # dummy_or_subproc = "dummy"
# dummy_or_subproc = "subproc"
# seed=1
# # seed=None
# n_quant = 200
# # kappa=1.0
# kappa=0.01 # should be 0, but torch does not support 0.


### Dabney's setting (2017)
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
SAVE_PATH = './QRDQN_model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/QRDQN_vanilla'
LOG_INTERVAL = 1000
use_cuda = True
# use_cuda = False
# dummy_or_subproc = "dummy"
dummy_or_subproc = "subproc"
n_quant = 200
# kappa=1.0
kappa=0.01
LR = 5e-5 
min_sq_grad = 0.01 / BATCH_SIZE # epsilon_adam
# min_sq_grad = None # epsilon_adam
seed=1
# seed=None


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

    if seed!=None: # seed successfully gives us reproducible results, but resutls may be different between cpu and cuda.
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
    online_net = Network(env, device=device, n_quant=n_quant) 
    target_net = Network(env, device=device, n_quant=n_quant)
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
            replay_buffer.append(transition) # 4 agents play in their own scenarios and collect their own data.

            if done:
                epinfos_buffer.append(info['episode'])
                episode_count += 1

        obses = new_obses

        ## Start Gradient Descent
        transitions = random.sample(replay_buffer, BATCH_SIZE) 
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













