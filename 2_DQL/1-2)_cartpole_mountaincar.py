### https://www.youtube.com/watch?v=NP8pXZdU-5U&list=LL&index=1&t=1435s  => youtube lecture
### https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py  => description of the setting in CartPole (e.g.) state & action space
### https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py  => description of the setting in MountainCar (e.g.) state & action space
### recommended to install gym=0.17.3 (pip install gym=0.17.3). Otherwise, there becomes extra {} in obs when a new episode begins.


##### STEP 1: IMPORT LIBRARIES

import torch
from torch import nn
import numpy as np
import random
import gym
from collections import deque
import itertools


##### STEP 2: TUNING PARAMETERS

# gamma=0.99
gamma=0.999

batch_size=32
min_replay_size=1000 # when |D| gets bigger than this, start training.
buffer_size=50000 # maximum level of |D|

epsilon_start=1.0
epsilon_end=0.02
epsilon_decay=10000

target_update_freq=1000

max_step = 100000

# env=gym.make('CartPole-v1')
env=gym.make('MountainCar-v0')


##### STEP 3: CREATE NETWORK

### 3.1: nn.Module

class Network(nn.Module):
    def __init__(self, env): # => online_net = Network(env)
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape)) # state-space is 4-dimension, because it considers 4 different properties (position, velocity, angle, angular velocity). 
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), # model the network so that the inputs are as many as the features.
            nn.Tanh(),
            nn.Linear(64, env.action_space.n) # model the network so that there are as many outputs as possible actions.
        )

    def forward(self, x): # => online_net(obses_t)
        return self.net(x)
    
    # # youtuber's code
    # def act(self, obs): # => online_net.act(obs)
    #     obs_t = torch.as_tensor(obs, dtype=torch.float32)
    #     q_values = self(obs_t.unsqueeze(0)) # for each state, there can be two actions.
    #     max_q_index = torch.argmax(q_values, dim=1)[0] # select the maximizing action. 
    #     action = max_q_index.detach().item()
    #     return action

    # my code
    def act(self, obs): # => online_net.act(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t) # for each state, there can be two actions.
        max_q_index = torch.argmax(q_values) # select the maximizing action. 
        action = max_q_index.item()
        return action

### 3.2: NETWORK

replay_buffer=deque(maxlen=buffer_size) # The size of our replay_buffer D is not going to exceed buffer_sizer. (|D|<=buffer_size) => merits of deque.
return_buffer=deque([0.0], maxlen=100)  # We are only saving the most recent 100 returns. 
episode_return=0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict()) # Synchronize target-network = online-network.

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4) # Written here! We are updating the parameters of the online network, not the target network!


##### STEP 4: INITIALIZE REPLAY BUFFER

obs=env.reset()

for _ in range(min_replay_size): # We will first collect samples as many as min_replay_size (form the set D), and then begin learning.
    
    action = env.action_space.sample() # integer (not float)
    new_obs, rew, done, _ = env.step(action) # s_prime, reward, whether terminal or not. The last two ones are unnecessary.
    transition = (obs, action, rew, done, new_obs) # We store this as a tuple.
    replay_buffer.append(transition) # We are collecting D so that we can start learning.
    obs=new_obs

    if done:
        obs=env.reset() # If we reached the terminal state, we restart the episode.


##### STEP 5: TRAINING LOOP

obs=env.reset()
step=0

iter=0 # If we want to repeat this traning loop, start from here.
for step in itertools.count(start=step): # basically same with while True -> may need some break if we want to stop.

    # step=1

    ## (Part1) One step of action by epsilon-greedy policy

    epsilon = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end]) 
    # From a line that connects (0, epsilon_start) and (epsilon_decay, epsilon_end), we interpolate. 
    # Gives us the bound values when x-value is out of the bound (outside [0, epsilon_decay]).
    rnd_sample = random.random() # can be understood as uniform(0,1)

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action= online_net.act(obs)

    new_obs, rew, done, _ = env.step(action)
    # a=env.render() # may comment out if we want to speed up the training by not showing the animation.
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition) # plug new tuple of (s, a, reward, s_prime) into our replay_buffer D.

    obs=new_obs
    episode_return += rew  
    # Of course this is not the actual return that our learning (gamma=0.99) is targetting, since this is assuming gamma=1.0 in this case.
    # However, if we learn with gamma=1, it will be more prone to diverge, so we should use gamma=0.99 when learning.
    # Nevertheless, what we are really interested in is how long the agent persists, so we evaluate it with gamma=1.0.
    
    if done:
        obs=env.reset()
        return_buffer.append(episode_return) # Every time we are done with an episode, we obtain one return. (but with limit of return_buffer size)
        episode_return=0


    ## (Part2) Sample Batch

    transitions = random.sample(replay_buffer, batch_size)

    obses = np.asarray([t[0] for t in transitions]) # We do => np-array => torch-tensor instead of directly => torch-tensor, because this way it is faster.
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions]) 
    
    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    ## (Part3) Compute Targets, Loss, and Gradient Descent

    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # [0]: maximum values / [1]: argmax
    targets = rews_t + gamma * (1-dones_t) * max_target_q_values
    
    q_values = online_net(obses_t) 
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) # for each row of the matrix q_values, pick the index=actions_t.
    loss=nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # We are updating the online network, not the target network. (defined in optimizer)

    ## (Part4) Update Target Network

    if step % target_update_freq==0:
        target_net.load_state_dict(online_net.state_dict()) # Synchronize target network with online network.

    ## (Part5) Logging

    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Return', np.mean(return_buffer))

    iter += 1
    if iter-1==max_step:
        break


##### STEP 6: DISPLAY THE TRAINED MODEL

while True: # Stop with Ctrl+C. When doing it, we need to click the terminal, and no part should be dragged (highlighted). Then Ctrl+C works.
    action=online_net.act(obs)
    obs, _, done, _ = env.step(action)
    a=env.render()
    if done:
        obs=env.reset()

