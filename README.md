# Reinforcement Learning (RL) Tutorial
Machine learning mainly consists of three methods: Supervised Learning, Unsupervised Learning and Reinforcement Learning. Supervised Learning provides mapping functionality between input and output using labelled dataset. Some of the supervised learning methods: Linear Regression, Support Vector Machines, Neural Networks, etc. Unsupervised Learning provides grouping and clustering functionality. Some of the supervised learning methods: K-Means, DBScan, etc. Reinforcement Learning is different from supervised and unsupervised learning. RL provides behaviour learning. 

**Keywords:** Dynamic Programming (Policy and Value Iteration), Monte Carlo, Temporal Difference (SARSA, QLearning), Approximation, Policy Gradient, DQN.

**NOTE: There are many tutorials in the internet. This one summarizes all of the tutorials with demo of RL algorithms.**

## What is RL?
"A reinforcement learning algorithm, or agent, learns by interacting with its environment. The agent receives rewards by performing correctly and penalties for performing incorrectly. The agent learns without intervention from a human by maximizing its reward and minimizing its penalty" [*](https://www.techopedia.com/definition/32055/reinforcement-learning). RL agents are used in different applications: Robotics, self driving cars, playing atari games, managing investment portfolio, control problems. I am believing that like many AI laboratories do, reinforcement learning with deep learning will be a core technology in the future.


![rl-agent-env](https://user-images.githubusercontent.com/10358317/49733000-71881000-fc91-11e8-89ab-503775f44d32.jpg) 
[Sutton & Barto Book: RL: An Introduction]

### Markov Decision Process:
- It consists of five tuples: status, actions, rewards, state transition probability, discount factor.
- Markov decision processes formally describe an environment for reinforcement learning.
- ![mdps](https://user-images.githubusercontent.com/10358317/49738227-e95d3700-fc9f-11e8-8ad8-cec2267d4668.jpg) [David Silver Lecture Notes]


#### Markov Property:
A state S<sub>t</sub> is Markov if and only if P[S<sub>t+1</sub>|S<sub>t</sub>] =P[S<sub>t+1</sub>|S<sub>1</sub>,...,S<sub>t</sub>]


## RL Components:
### Rewards:
- A reward Rt is a scalar feedback signal.
- The agent’s job is to maximise cumulative reward
### State Transition Probability:
- p(s′,r|s,a).=  Pr{S<sub>t</sub>=s′,R<sub>t</sub>=r|S<sub>t-1</sub>=s,A<sub>t−1</sub>=a},
### Discount Factor:
- The discount γ∈[0,1] is the present value of future rewards.
### Return:
- The return G<sub>t</sub> is the total discounted reward from time-step t.

![return](https://user-images.githubusercontent.com/10358317/49737029-c9784400-fc9c-11e8-8e05-23e6d7bb9fd0.jpg)
[David Silver Lecture Notes]
### Value Function:
- Value function is a prediction of future reward. How good is each state and/or action.
- The value function v(s) gives the long-term value of state s
- V<sub>π</sub>(s) =E<sub>π</sub>[R<sub>t+1</sub>+γR<sub>t+2</sub>+γ<sup>2</sup>R<sub>t+3</sub>+...|S<sub>t</sub>=s]
- Value function has two parts: immediate reward and discounted value of successor state.

![value_function](https://user-images.githubusercontent.com/10358317/49737276-7eaafc00-fc9d-11e8-83ad-e21feec25c16.jpg)
[David Silver Lecture Notes]
### Policy (π): 
A policy is the agent’s behaviour. It is a map from state to action. 
- Deterministic policy: a=π(s).
- Stochastic policy: π(a|s) =P[A<sub>t</sub>=a|S<sub>t</sub>=s].
### State-Value Function:
![state-value](https://user-images.githubusercontent.com/10358317/49737548-3b9d5880-fc9e-11e8-8549-d868556f0569.jpg)
[David Silver Lecture Notes]
### Action-Value Function: 
![action- value](https://user-images.githubusercontent.com/10358317/49737562-448e2a00-fc9e-11e8-9e57-8c04649b9a99.jpg)
[David Silver Lecture Notes]

### Optimal Value Functions:
![optimal-value](https://user-images.githubusercontent.com/10358317/49737868-f3cb0100-fc9e-11e8-82f8-f718f2af6b51.jpg)
[David Silver Lecture Notes]

### Planning vs RL:
#### Planning:
- Rules of the game are known.
- A model of the environment is known.
- The agent performs computations with its mode.
- The agent improves its policy.
#### RL:
- The environment is initially unknown.
- The agent interacts with the environment.
- The agent improves its policy.

### Exploration and Exploitation:
- Reinforcement learning is like trial-and-error learning.
- The agent should discover a good policy.
- Exploration finds more information about the environment.
- Exploitation exploits known information to maximise reward

### Prediction & Control Problem (Pattern of RL algorithms):
- Prediction: evaluate the future (Finding value given a policy).
- Control: optimise the future (Finding optimal/best policy).

## Grid World:
- Grid World is a game for demonstration. 12 positions, 11 states, 4 actions. Our aim is to find optimal policy. 

![grid-world](https://user-images.githubusercontent.com/10358317/49739821-d77d9300-fca3-11e8-893d-af7690d114b2.jpg)

![optimal-policy-grid](https://user-images.githubusercontent.com/10358317/49739871-f714bb80-fca3-11e8-93c3-43a72284b7ba.jpg)

## Dynamic Programming Method (DP): Full Model
- Dynamic Programming is a very general solution method for problems which have two properties:
1.Optimal substructure, 2.Overlapping subproblems.
- Markov decision processes satisfy both properties. Bellman equation gives recursive decomposition. Value function stores and reuses solutions.
- In DP method, full model is known, It is used for planning in an MDP.
- There are 2 methods: Policy Iteration, Value Iteration.
- DP uses full-width backups.
- DP is effective for medium-sized problems (millions of states).
- For large problems DP suffers Bellman’s curse of dimensionality. 

### Policy Iteration (with Pseudocode):
- Policy Iteration consists of 2 main step: 1.Policy Evaluation, 2.Policy Iteration.

![policy iteration](https://user-images.githubusercontent.com/10358317/49740185-bf5a4380-fca4-11e8-80a4-fdc8dce8e349.jpg) [David Silver Lecture Notes]

![policy-iteration](https://user-images.githubusercontent.com/10358317/49804775-871a3a00-fd64-11e8-90da-9550bcd1175c.jpg)


#### Policy Evaluation (with Pseudocode):
- Problem:  evaluate a given policy π.
- Solution: iterative application of Bellman expectation backup.
- v1 → v2→ ... → vπ.

![iterative-policy-evaluation](https://user-images.githubusercontent.com/10358317/49739932-1d3a5b80-fca4-11e8-962f-26348b323c63.jpg)

#### Policy Improvement:
![policy-improvement](https://user-images.githubusercontent.com/10358317/49804622-14a95a00-fd64-11e8-9ae0-932af77bbc0c.jpg)

### Value Iteration (with Pseudocode):
- Policy iteration has 2 inner loop. However, value iteration has a better solution.
- It combines policy evaluation and policy improvement into one step.
- Problem:  find optimal policy π.
- Solution: iterative application of Bellman optimality backup.

![value-iteration](https://user-images.githubusercontent.com/10358317/49805004-1de6f680-fd65-11e8-95b8-a3c224fbbe53.jpg)

## Monte Carlo (MC) Method:
- MC methods learn directly from episodes of experience.
- MC is model-free :  no knowledge of MDP transitions / rewards.
- MC uses the simplest possible idea: value = mean return.
- Episode must terminate before calculating return.
- Average return is calculated instead of using true return G.
- First Visit MC: The first time-step t that state s is visited in an episode.
- Every Visit MC: Every time-step t that state s is visited in an episode.

### MC Calculating Returns (with Pseudocode):
![mc-calculating-returns](https://user-images.githubusercontent.com/10358317/49827998-cca62980-fd9b-11e8-999b-150aac525870.jpg)

### First-Visit MC (with Pseudocode) (MC Prediction Problem):
![first-visit-mc](https://user-images.githubusercontent.com/10358317/49827884-73d69100-fd9b-11e8-9623-16890aa3bbcb.jpg)

### Exploring-Starts (with Pseudocode) (MC Control Problem):
- State s and Action a is randomly selected for all starting points.
- Use Q instead of V 
- Update the policy after every episode, keep updating the same Q in-place.

![mc-control1](https://user-images.githubusercontent.com/10358317/49828847-fbbd9a80-fd9d-11e8-9286-dee68c6fa1a2.jpg)

### MC Epsilon Greedy (without Exploring Starts):
- MC Exploring start is infeasible, because in real problems we can not calculate all edge cases (ex: in self-driving car problem, we can not calculate all cases).
- Randomly selection for all starting points in code is removed.
- Change policy to sometimes be random.
- This random policy is Epsilon-Greedy (like multi-armed bandit problem)

## Temporal Difference (TD) Learning Method:
- TD methods learn directly from episodes of experience.
- TD updates a guess towards a guess
- TD learns from incomplete episodes, by bootstrapping.
- TD uses bootstrapping like DP, TD learns experience like MC.

## MC - TD Difference:
- TD can learn before knowing the final outcome.
- TD can learn online after every step. MC must wait until end of episode before return is known.
- TD can learn without the final outcome.
- TD can learn from incomplete sequences. MC can only learn from complete sequences.
- TD works in continuing environments. MC only works for episodic environments.
- MC has high variance, zero bias. TD has low variance, some bias.

![mc-td-dif1](https://user-images.githubusercontent.com/10358317/49805899-60a9ce00-fd67-11e8-900e-38662cf36a54.jpg)
![mc-td-dif2](https://user-images.githubusercontent.com/10358317/49805902-61dafb00-fd67-11e8-8033-b06f8a3ed1c1.jpg)
![mc-td-dif3](https://user-images.githubusercontent.com/10358317/49810084-758b5f00-fd71-11e8-8b67-b1d8da52e45a.jpg)
[David Silver Lecture Notes]

## MC - TD - DP Difference in Visual:
![mc-td-dp](https://user-images.githubusercontent.com/10358317/49806522-01e55400-fd69-11e8-92a6-9bff14bb4c80.jpg)
[David Silver Lecture Notes]

### SARSA:
### Q-Learning:

## Function Approximation:

## Open AI Gym:

## Policy Gradient:
- Model Free Algorithm.

## Actor-Critic:
- Model Free Algorithm.

## Deep Q Learning:

## Imitation Learning:

## Meta-Learning:

## Inverse RL: 

## Deep RL: 

## POMDPs (Partial Observable MDP): 
![pomdps](https://user-images.githubusercontent.com/10358317/49738014-5c19e280-fc9f-11e8-8ca6-fe8fbeb0a5df.jpg)
[David Silver Lecture Notes]


## Resources:
Free Lectures:
http://rail.eecs.berkeley.edu/deeprlcourse/

https://sites.google.com/view/deep-rl-bootcamp/lectures



Udemy:

Udacity:

## Papers:
- Q-Learning: V.  Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, et al. “Playing Atari with Deep Reinforcement Learning”. (2013).
- Policy Gradient: J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P.  Abbeel. “Trust Region Policy Optimization”. (2015).
- Policy Gradient: V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, et al. “Asynchronous methods for deep reinforcement learning”. (2016).
- Robots-Guided policy search: S. Levine*, C. Finn*, T. Darrell, P. Abbeel. “End-to-end training of deep visuomotor policies”. (2015).
- Robots-Q-Learning: D. Kalashnikov et al. “QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation”. (2018)
- AlphaGo- Supervised learning + policy gradients + value functions + Monte Carlo tree search D. Silver, A. Huang, C. J.Maddison, A. Guez, L. Sifre, et al. “Mastering the game of Go with deep neural networks and tree search”. Nature (2016).


## References:
- Sutton & Barto Book: Reinforcement Learning: An Introduction
- David Silver Lecture Notes: (http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
