# Report

### Solution Method

For this problem we will use deep deterministic policy gradient (DDPG) algorithm [1]. Given our environment choice, we will have 20 parallel agents working simultaneously.

The actor network has two hidden layers, both fully connected and with 400, and 300 neurons respectively. Hidden layers use ReLU as activation function; the output layers uses hyperbolic tangent instead.
The critic network has a similar architecture to the actor network, with the main difference in the output layer, consisting of one single value.

Both networks' losses are minimized via Adam optimizer.

Hyperparameters are set as follows:<br/>
replay buffer size = 100000<br/>
batch size = 128<br/>
discount factor = 0.99<br/>  
tau = .001<br/>
actor learning rate = .0001<br/>  
critic learning rate = .001<br/>  
L2 weight decay = 0<br/>



### Results
This environment is considered solved after the agents achieve an average score (across all the agenets) of +30 over 100 episodes. 
With our implementation and hyperparameters choice, the agents achieve this in 74 episodes (with random seed=0). 
Here a plot of the score development through the episodes.

![Scores](img/scores.png)

The training took approximately 28 minutes with an Intel i7-8700K 3.70GHz. 

### Future work and possible improvements
Our goal of +30 score was met via a solution similar to Udacity's pendulum DDPG implementation. Nevertheless, it would be worth to investigate different network architectures, and to perform a more thorough hyperparameter selection in order to achieve a higher score, or the same score in less iterations.
Moreover, other algorithms such as proximal policy optimization (PPO) [2] could be implemented and compared with DDPG.


### Literature
[1] <a href="https://arxiv.org/pdf/1509.02971.pdf" target="_blank">Continuous control with deep reinforcement learning</a><br/>
[1] <a href="https://arxiv.org/pdf/1707.06347.pdf" target="_blank">Proximal Policy Optimization Algorithms</a>

