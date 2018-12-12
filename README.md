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
- There are 3 techniques for solving MDPs: Dynamic Programming (DP) Learning, Monte Carlo (MC) Learning, Temporal Difference (TD) Learning.

![mdps](https://user-images.githubusercontent.com/10358317/49738227-e95d3700-fc9f-11e8-8ad8-cec2267d4668.jpg) [David Silver Lecture Notes]


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
- Disadvantage of DP: requires full model of environment, never learns from experience.

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
- TD uses bootstrapping like DP, TD learns experience like MC (combines MC and DP).

## MC - TD Difference:
- MC and TD learn from experience.
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

### SARSA (TD Control Problem, On-Policy):
- In on-policy learning the Q(s,a) function is learned from actions, we took using our current policy π.

![updatingwithsarsa](https://user-images.githubusercontent.com/10358317/49831282-8c977480-fda4-11e8-8c7b-473ad5040f9d.jpg)
![sarsa-algo](https://user-images.githubusercontent.com/10358317/49831108-23affc80-fda4-11e8-84ca-08c6f1c056c5.jpg)



### Q-Learning (TD Control Problem, Off-Policy):
- Looks like SARSA, instead of choosing a' based on argmax of Q, Q(s,a) is updated directly with max over  Q(s',a')
- In off-policy learning the Q(s,a) function is learned from different actions (for example, random actions). We even don't need a policy at all.

![qfunction](https://user-images.githubusercontent.com/10358317/49831386-d41e0080-fda4-11e8-967e-dd184a4e07e5.jpg)

![updatingwithqlearning](https://user-images.githubusercontent.com/10358317/49831118-26aaed00-fda4-11e8-9420-0ba120b1a509.jpg)
![qlearning-algo](https://user-images.githubusercontent.com/10358317/49831121-29a5dd80-fda4-11e8-9a72-aee5c9781950.jpg)

## Function Approximation:
- Reinforcement learning can be used to solve large problems, e.g Backgammon:  10<sup>20</sup> states; Computer Go:  10<sup>170</sup> states; Helicopter: continuous state space.
- So far we have represented value function by a lookup table, Every state s has an entry V(s) or every state-action pair s,
a has an entry Q(s,a).
- There are too many states and/or actions to store in memory. It is too slow to learn the value of each state individually. Tabulated Q may not fit memory.
- Solution for large MDPs:

![func-appr](https://user-images.githubusercontent.com/10358317/49852371-fb042300-fdf4-11e8-8d15-1b77eb1c2fda.jpg)

- Differentiable function approximators can be used: Linear combinations of features, Neural Networks.

![value-func-appr](https://user-images.githubusercontent.com/10358317/49852598-a3b28280-fdf5-11e8-8a6d-c958136ca744.jpg)

![func-appr2](https://user-images.githubusercontent.com/10358317/49852876-66022980-fdf6-11e8-9820-380c39b280d0.jpg)

### Feature Vector:
![feature-vectors](https://user-images.githubusercontent.com/10358317/49852662-d2c8f400-fdf5-11e8-9550-5cf87360e964.jpg)

## Open AI Gym Environment:

## Policy Gradient:
- DP, MC and TD Learning methods are value-based methods (Learnt Value Function, Implicit policy).
- In value-based methods, a policy was generated directly from the value function (e.g. using epsilon-greedy)
- In policy-based, we will directly parametrise the policy ( π<sub>θ</sub>(s,a) =P[a|s,θ) ).
- Policy Gradient method is a policy-based method (No Value Function, Learnt Policy).

![policy-based](https://user-images.githubusercontent.com/10358317/49868713-47fcef00-fe1f-11e8-97d8-cb0b15f2c3eb.jpg)

Advantages: 
- Better convergence properties,
- Effective in high-dimensional or continuous action spaces,
- Can learn stochastic policies.

Disadvantages:
- Typically converge to a local rather than global optimum.
- Evaluating a policy is typically inefficient and high variance.

### Policy Objective Functions:
- Policy based reinforcement learning is an optimisation problem.
- Find θ that maximises J(θ).

![policy-objective-func](https://user-images.githubusercontent.com/10358317/49869176-a7a7ca00-fe20-11e8-8152-07a3ae3d00d7.jpg)

### Policy-Gradient:

![policy-gradient](https://user-images.githubusercontent.com/10358317/49869270-ffdecc00-fe20-11e8-9e2a-2811a3e1ecf0.jpg)

![score-function](https://user-images.githubusercontent.com/10358317/49869466-80053180-fe21-11e8-81b5-2cd4ef740609.jpg)

![policy-gradient-theorem](https://user-images.githubusercontent.com/10358317/49869560-cb1f4480-fe21-11e8-87eb-7ce93930038b.jpg)

### Monte-Carlo Policy Gradient (REINFORCE):
![reinforce](https://user-images.githubusercontent.com/10358317/49869620-f1dd7b00-fe21-11e8-8023-a3de70e2cbbb.jpg)

## Actor-Critic:
- Actor-Critic method is a policy-based method (Learnt Value Function, Learnt Policy).

## Deep Q Learning (Deep Q-Networks: DQN):
- Gradient descent is simple and appealing. But it is not sample efficient.
- Batch methods seek to find the best fitting value function.
- Given the agent’s experience (“training data”)

### Experience Replay:
![dqn-experience-replay](https://user-images.githubusercontent.com/10358317/49853044-fa6c8c00-fdf6-11e8-94be-27a32fad21a2.jpg)

### DQN in Atari:
-V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, et al. “Playing Atari with Deep Reinforcement Learning”. (2013)

![dqn-in-atari](https://user-images.githubusercontent.com/10358317/49853048-fb9db900-fdf6-11e8-862d-d93acddceecf.jpg)

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
