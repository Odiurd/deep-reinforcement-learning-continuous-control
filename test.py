from unityagents import UnityEnvironment
from ddpg_agent import Agent
import torch
import numpy as np


ENV_PATH = "E:/Users/Megaport/Dropbox/Python/Udacity/Project 2/Reacher_Windows_x86_64/Reacher.exe"
ACTOR_CHECKPOINT_NAME = 'checkpoint_actor.pth'
CRITIC_CHECKPOINT_NAME = 'checkpoint_critic.pth'
GRAPHICS_OFF = False

n_episodes = 3

env = UnityEnvironment(file_name=ENV_PATH, no_graphics=GRAPHICS_OFF)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

agents = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)
agents.actor_local.load_state_dict(torch.load("ckpt/{}".format(ACTOR_CHECKPOINT_NAME)))
agents.critic_local.load_state_dict(torch.load("ckpt/{}".format(CRITIC_CHECKPOINT_NAME)))

for i_episode in range(1, n_episodes+1):
    print('Starting episode {}'.format(i_episode))
    env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name] 
    state = env_info.vector_observations
    agents.reset()
    score = np.zeros(num_agents)
    while True:
        action = agents.act(state)
            
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
            
        agents.step(state, action, rewards, next_state, dones)
        state = next_state
        score += rewards
                
        if np.any(dones):
            print('Score: {}'.format(np.mean(score)))
            break 