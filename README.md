# Reinforcement Learning (RL) Tutorial
Machine learning mainly consists of three methods: Supervised Learning, Unsupervised Learning and Reinforcement Learning. Supervised Learning provides mapping functionality between input and output using labelled dataset. Some of the supervised learning methods: Linear Regression, Support Vector Machines, Neural Networks, etc. Unsupervised Learning provides grouping and clustering functionality. Some of the supervised learning methods: K-Means, DBScan, etc. Reinforcement Learning is different from supervised and unsupervised learning. RL provides behaviour learning. 

**Keywords:** Dynamic Programming (Policy and Value Iteration), Monte Carlo, Temporal Difference (SARSA, QLearning), Approximation, Policy Gradient, DQN.

## What is RL?
"A reinforcement learning algorithm, or agent, learns by interacting with its environment. The agent receives rewards by performing correctly and penalties for performing incorrectly. The agent learns without intervention from a human by maximizing its reward and minimizing its penalty" [*](https://www.techopedia.com/definition/32055/reinforcement-learning). RL agents are used in different applications: Robotics, self driving cars, playing atari games, managing investment portfolio, control problems. I am believing that like many AI laboratories do, reinforcement learning with deep learning will be a core technology in the future.

![rl-agent-env](https://user-images.githubusercontent.com/10358317/49733000-71881000-fc91-11e8-89ab-503775f44d32.jpg) 
[Sutton & Barto Book: RL: An Introduction]

## Multi-armed Bandit Problem:

## RL Components:
### Markov Decision Process:
### Status:
### Actions:
### Rewards:
### Trans. Probability:
### Discount Factor:
### Value Function:
Value function is a prediction of future reward. How good is each state and/or action.
- V<sub>π</sub>(s) =E<sub>π</sub>[R<sub>t+1</sub>+γR<sub>t+2</sub>+γ<sup>2</sup>R<sub>t+3</sub>+...|S<sub>t</sub>=s]
### Policy (π): 
A policy is the agent’s behaviour. It is a map from state to action. 
- Deterministic policy: a=π(s).
- Stochastic policy: π(a|s) =P[A<sub>t</sub>=a|S<sub>t</sub>=s].
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

### Prediction & Control Problem:
- Prediction: evaluate the future (Given a policy).
- Control: optimise the future (Find the best policy).

## Grid World:

## Dynamic Programming Method: Full Model
### Policy Iteration:
### Iterative policy evaluation:
### Policy Improvement:

## Monte Carlo Method:
### Exploring Stars:
### Monte Carlo Prediction Problem:
### Monte Carlo Control Problem:
### Monte Carlo Epsilon Greedy:

## Temporal Difference Learning Method:
### TD(0) Learning
### SARSA:
### Q-Learning:

## Function Approximation:

## Open AI Gym:

## Policy Gradient:

## Actor-Critic

## Deep Q Learning:

## Imitation Learning:

## Meta-Learning:

## Inverse RL: 

## Deep RL: 

## Resources:
Free Lectures:
http://rail.eecs.berkeley.edu/deeprlcourse/

https://sites.google.com/view/deep-rl-bootcamp/lectures

http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

Udemy:

Udacity:

## Papers:
Playing Atari
AlphaGo, AlphaGo Zero
Imitation Learning Paper

## References:
Sutton & Barto Book: Reinforcement Learning: An Introduction PDF
