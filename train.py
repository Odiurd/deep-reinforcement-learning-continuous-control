from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_agent import Agent
import torch
import matplotlib.pyplot as plt
from time import strftime, gmtime


ENV_PATH = "E:/Users/Megaport/Dropbox/Python/Udacity/Project 2/Reacher_Windows_x86_64/Reacher.exe"
ACTOR_CHECKPOINT_NAME = 'checkpoint_actor.pth'
CRITIC_CHECKPOINT_NAME = 'checkpoint_critic.pth'
IMAGE_NAME = 'scores.png'
TARGET_SCORE = 30
GRAPHICS_OFF = True


def plot(scores, IMAGE_NAME):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('img/{}'.format(IMAGE_NAME))    
    
    
def ddpg(n_episodes=2000, store_every=10):
    scores_deque = deque(maxlen=store_every)
    scores = []
    
    agents = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)
    
    for i_episode in range(1, n_episodes+1):
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
                break 
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        avg_score = np.mean(scores_deque)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\t {}'.format(i_episode,
                                                                                np.mean(scores_deque), np.mean(score),
                                                                                strftime("%H:%M:%S", gmtime())), end="")         
        if i_episode % store_every == 0 or avg_score >= TARGET_SCORE:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            
            if avg_score >= TARGET_SCORE:
                torch.save(agents.actor_local.state_dict(), "ckpt/{}".format(ACTOR_CHECKPOINT_NAME))
                torch.save(agents.critic_local.state_dict(), "ckpt/{}".format(CRITIC_CHECKPOINT_NAME)) 
                break
            
    return scores  


env = UnityEnvironment(file_name=ENV_PATH, no_graphics=GRAPHICS_OFF)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

print('Number of agents: {}'.format(num_agents))
print('Number of actions: {}'.format(action_size))
print('Number of states: {}'.format(state_size))

print('First state: {}'.format(states[0]))


if torch.cuda.is_available():
    print("trainining on GPU")
else:
    print("training on CPU")
    
print('Training start time: {}'.format(strftime("%H:%M:%S", gmtime())))

scores_tot = ddpg()
plot(scores_tot, IMAGE_NAME)
env.close()

print('\nTraining end time: {}'.format(strftime("%H:%M:%S", gmtime())))
