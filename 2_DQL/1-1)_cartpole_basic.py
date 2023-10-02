import gym
import random

env = gym.make("CartPole-v0", render_mode='rgb_array')

for episode in range(50): # originally 10
    # episode=0
        env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        for t in range(10000): # originally 500
            # t=0

            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()
            

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)
            
            # lets print everything in one line:
            print(t, next_state, reward, done, info, action)
            if done:
                break
