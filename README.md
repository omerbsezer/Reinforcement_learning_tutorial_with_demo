# Reinforcement Learning (RL) Tutorial
There are many RL tutorials, courses, papers in the internet. This one summarizes all of the RL tutorials, RL courses, and some of the important RL papers including sample code of RL algorithms. It will continue to be updated over time.

**Keywords:** Dynamic Programming (Policy and Value Iteration), Monte Carlo, Temporal Difference (SARSA, QLearning), Approximation, Policy Gradient, DQN, Imitation Learning, Meta-Learning, RL papers, RL courses, etc.

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table of Contents
- [What is Reinforcement Learning?](#whatisRL)
- [Markov Decision Process](#mdp)
    - [Markov Property](#markovproperty)
- [RL Components](#RLcomponents)
    - [Rewards](#Rewards)
    - [State Transition Probability](#StateTransitionProbability)
    - [Discount Factor](#DiscountFactor)
    - [Return](#Return)
    - [Value Function](#ValueFunction)
    - [Policy](#Policy)
    - [State-Value Function](#StateValueFunction)
    - [Action-Value Function](#ActionValueFunction)
    - [Planning vs RL](#PlanningRL)
    - [Exploration and Exploitation](#ExplorationandExploitation)
    - [Prediction & Control Problem](#PredictionControlProblem)
- [Grid World](#GridWorld)
- [Dynamic Programming Method (DP)](#DP)
    - [Policy Iteration](#PolicyIteration)
        - [Policy Evaluation](#PolicyEvaluation)
        - [Policy Improvement](#PolicyImprovement)
    - [Value Iteration](#ValueIteration)
 - [Monte Carlo (MC) Method](#MonteCarlo)
    - [MC Calculating Returns](#MCCalculatingReturns)
    - [First-Visit MC](#FirstVisitMC)
    - [MC Exploring-Starts](#MCExploringStarts)
    - [MC Epsilon Greedy](#MCEpsilonGreedy)
  - [Temporal Difference (TD) Learning Method](#TDLearning)
    - [MC - TD Difference](#MCTDDifference)
    - [MC - TD - DP Difference in Visual](#MCTDDifferenceinVisual)
    - [SARSA (TD Control Problem, On-Policy)](#SARSA)
    - [Q-Learning (TD Control Problem, Off-Policy)](#Qlearning)
  - [Function Approximation](#FunctionApproximation)
    - [Feature Vector](#FeatureVector)
  - [Open AI Gym Environment](#OpenAIGym)
  - [Policy-Based Methods](#PolicyBased)
    - [Policy Objective Functions](#PolicyObjectiveFunctions)
    - [Policy-Gradient](PolicyGradient)
    - [Monte-Carlo Policy Gradient (REINFORCE)](#REINFORCE)
  - [Actor-Critic](#ActorCritic)
    - [Action-Value Actor-Critic](#ActorValueActorCritic)
    - [Actor-Critic Algorithm:A3C](#A3C)
    - [Different Policy Gradients](#DifferentPolicyGradients)
  - [Model-Based RL](ModelBasedRL)
    - [Real and Simulated Experience](#RealandSimulatedExperience)
    - [Dyna-Q Algorithm](#DynaQ)
    - [Sim-Based Search](#SimBased)
    - [MC-Tree-Search](#MCTreeSearch)
    - [Temporal-Difference Search](#TemporalDifferenceSearch)
    - [RL in Games](#RLinGames)
  - [Deep Q Learning (Deep Q-Networks: DQN)](#DQN)
    - [Experience Replay](#ExperienceReplay)
    - [DQN in Atari](#DQNAtari)
  - [Imitation Learning](#ImitationLearning)
    - [Dagger: Dataset Aggregation](#Dagger)
    - [PLATO: Policy Learning with Adaptive Trajectory Optimization](#PLATO)
    - [One-Shot Imitation Learning](#OneShotImitation)
  - [Meta-Learning](#MetaLearning)
  - [POMDPs (Partial Observable MDP)](#POMDPs)
  - [Resources](#Resources)
  - [Important RL Papers](#ImportantRLPapers)
  - [References](#References)
    
  
## What is RL? <a name="whatisRL"></a>
Machine learning mainly consists of three methods: Supervised Learning, Unsupervised Learning and Reinforcement Learning. Supervised Learning provides mapping functionality between input and output using labelled dataset. Some of the supervised learning methods: Linear Regression, Support Vector Machines, Neural Networks, etc. Unsupervised Learning provides grouping and clustering functionality. Some of the unsupervised learning methods: K-Means, DBScan, etc. Reinforcement Learning is different from supervised and unsupervised learning. RL provides behaviour learning. 

"A reinforcement learning algorithm, or agent, learns by interacting with its environment. The agent receives rewards by performing correctly and penalties for performing incorrectly. The agent learns without intervention from a human by maximizing its reward and minimizing its penalty" [*](https://www.techopedia.com/definition/32055/reinforcement-learning). RL agents are used in different applications: Robotics, self driving cars, playing atari games, managing investment portfolio, control problems. I am believing that like many AI laboratories do, reinforcement learning with deep learning will be a core technology in the future.


![rl-agent-env](https://user-images.githubusercontent.com/10358317/49733000-71881000-fc91-11e8-89ab-503775f44d32.jpg) 
[Sutton & Barto Book: RL: An Introduction]

## Markov Decision Process <a name="mdp"></a>:
- It consists of five tuples: status, actions, rewards, state transition probability, discount factor.
- Markov decision processes formally describe an environment for reinforcement learning.
- There are 3 techniques for solving MDPs: Dynamic Programming (DP) Learning, Monte Carlo (MC) Learning, Temporal Difference (TD) Learning.

![mdps](https://user-images.githubusercontent.com/10358317/49738227-e95d3700-fc9f-11e8-8ad8-cec2267d4668.jpg) [David Silver Lecture Notes]


### Markov Property <a name="markovproperty"></a>:
A state S<sub>t</sub> is Markov if and only if P[S<sub>t+1</sub>|S<sub>t</sub>] =P[S<sub>t+1</sub>|S<sub>1</sub>,...,S<sub>t</sub>]


## RL Components <a name="RLcomponents"></a>:
### Rewards <a name="Rewards"></a>:
- A reward Rt is a scalar feedback signal.
- The agent’s job is to maximise cumulative reward
### State Transition Probability <a name="StateTransitionProbability"></a>:
- p(s′,r|s,a).=  Pr{S<sub>t</sub>=s′,R<sub>t</sub>=r|S<sub>t-1</sub>=s,A<sub>t−1</sub>=a},
### Discount Factor <a name="DiscountFactor"></a>:
- The discount γ∈[0,1] is the present value of future rewards.
### Return <a name="Return"></a>:
- The return G<sub>t</sub> is the total discounted reward from time-step t.

![return](https://user-images.githubusercontent.com/10358317/49737029-c9784400-fc9c-11e8-8e05-23e6d7bb9fd0.jpg)
[David Silver Lecture Notes]

### Value Function <a name="ValueFunction"></a>:
- Value function is a prediction of future reward. How good is each state and/or action.
- The value function v(s) gives the long-term value of state s
- V<sub>π</sub>(s) =E<sub>π</sub>[R<sub>t+1</sub>+γR<sub>t+2</sub>+γ<sup>2</sup>R<sub>t+3</sub>+...|S<sub>t</sub>=s]
- Value function has two parts: immediate reward and discounted value of successor state.

![value_function](https://user-images.githubusercontent.com/10358317/49737276-7eaafc00-fc9d-11e8-83ad-e21feec25c16.jpg)
[David Silver Lecture Notes]

### Policy (π) <a name="Policy"></a>:
A policy is the agent’s behaviour. It is a map from state to action. 
- Deterministic policy: a=π(s).
- Stochastic policy: π(a|s) =P[A<sub>t</sub>=a|S<sub>t</sub>=s].
### State-Value Function <a name="StateValueFunction"></a>:
![state-value](https://user-images.githubusercontent.com/10358317/49737548-3b9d5880-fc9e-11e8-8549-d868556f0569.jpg)
[David Silver Lecture Notes]
### Action-Value Function <a name="ActionValueFunction"></a>:
![action- value](https://user-images.githubusercontent.com/10358317/49737562-448e2a00-fc9e-11e8-9e57-8c04649b9a99.jpg)
[David Silver Lecture Notes]

### Optimal Value Functions <a name="OptimalValueFunction"></a>:
![optimal-value](https://user-images.githubusercontent.com/10358317/49737868-f3cb0100-fc9e-11e8-82f8-f718f2af6b51.jpg)
[David Silver Lecture Notes]

### Planning vs RL <a name="PlanningRL"></a>:
#### Planning:
- Rules of the game are known.
- A model of the environment is known.
- The agent performs computations with its mode.
- The agent improves its policy.
#### RL:
- The environment is initially unknown.
- The agent interacts with the environment.
- The agent improves its policy.

### Exploration and Exploitation <a name="ExplorationandExploitation"></a>:
- Reinforcement learning is like trial-and-error learning.
- The agent should discover a good policy.
- Exploration finds more information about the environment (Gather more information).
- Exploitation exploits known information to maximise reward (Make the best decision given current information).

### Prediction & Control Problem (Pattern of RL algorithms) <a name="PredictionControlProblem"></a>:
- Prediction: evaluate the future (Finding value given a policy).
- Control: optimise the future (Finding optimal/best policy).

## Grid World <a name="GridWorld"></a>:
- Grid World is a game for demonstration. 12 positions, 11 states, 4 actions. Our aim is to find optimal policy.
- Demo Code: [gridWorldGame.py](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/gridWorldGame.py)

![grid-world](https://user-images.githubusercontent.com/10358317/49739821-d77d9300-fca3-11e8-893d-af7690d114b2.jpg)

![optimal-policy-grid](https://user-images.githubusercontent.com/10358317/49739871-f714bb80-fca3-11e8-93c3-43a72284b7ba.jpg)

## Dynamic Programming Method (DP): Full Model <a name="DP"></a>:
- Dynamic Programming is a very general solution method for problems which have two properties:
1.Optimal substructure, 2.Overlapping subproblems.
- Markov decision processes satisfy both properties. Bellman equation gives recursive decomposition. Value function stores and reuses solutions.
- In DP method, full model is known, It is used for planning in an MDP.
- There are 2 methods: Policy Iteration, Value Iteration.
- DP uses full-width backups.
- DP is effective for medium-sized problems (millions of states).
- For large problems DP suffers Bellman’s curse of dimensionality. 
- Disadvantage of DP: requires full model of environment, never learns from experience.

### Policy Iteration (with Pseudocode) <a name="PolicyIteration"></a>:
- Demo Code: [policy_iteration_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/policy_iteration_demo.ipynb)
- Policy Iteration consists of 2 main step: 1.Policy Evaluation, 2.Policy Iteration.

![policy iteration](https://user-images.githubusercontent.com/10358317/49740185-bf5a4380-fca4-11e8-80a4-fdc8dce8e349.jpg) [David Silver Lecture Notes]

![policy-iteration](https://user-images.githubusercontent.com/10358317/49804775-871a3a00-fd64-11e8-90da-9550bcd1175c.jpg)


#### Policy Evaluation (with Pseudocode) <a name="PolicyEvaluation"></a>:
- Problem:  evaluate a given policy π.
- Solution: iterative application of Bellman expectation backup.
- v1 → v2→ ... → vπ.

![iterative-policy-evaluation](https://user-images.githubusercontent.com/10358317/49739932-1d3a5b80-fca4-11e8-962f-26348b323c63.jpg)

#### Policy Improvement <a name="PolicyImprovement"></a>:
![policy-improvement](https://user-images.githubusercontent.com/10358317/49804622-14a95a00-fd64-11e8-9ae0-932af77bbc0c.jpg)

### Value Iteration (with Pseudocode) <a name="ValueIteration"></a>:
- Policy iteration has 2 inner loop. However, value iteration has a better solution.
- It combines policy evaluation and policy improvement into one step.
- Problem:  find optimal policy π.
- Solution: iterative application of Bellman optimality backup.

![value-iteration](https://user-images.githubusercontent.com/10358317/49805004-1de6f680-fd65-11e8-95b8-a3c224fbbe53.jpg)

## Monte Carlo (MC) Method <a name="MonteCarlo"></a>:
- Demo Code: [monte_carlo_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_demo.ipynb)
- MC methods learn directly from episodes of experience.
- MC is model-free :  no knowledge of MDP transitions / rewards.
- MC uses the simplest possible idea: value = mean return.
- Episode must terminate before calculating return.
- Average return is calculated instead of using true return G.
- First Visit MC: The first time-step t that state s is visited in an episode.
- Every Visit MC: Every time-step t that state s is visited in an episode.


### MC Calculating Returns (with Pseudocode) <a name="MCCalculatingReturns"></a>:
![mc-calculating-returns](https://user-images.githubusercontent.com/10358317/49827998-cca62980-fd9b-11e8-999b-150aac525870.jpg)

### First-Visit MC (with Pseudocode) (MC Prediction Problem) <a name="FirstVisitMC"></a>:
![first-visit-mc](https://user-images.githubusercontent.com/10358317/49827884-73d69100-fd9b-11e8-9623-16890aa3bbcb.jpg)

### MC Exploring-Starts (with Pseudocode) (MC Control Problem) <a name="MCExploringStarts"></a>:
- Demo Code: [monte_carlo_es_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_es_demo.ipynb)
- State s and Action a is randomly selected for all starting points.
- Use Q instead of V 
- Update the policy after every episode, keep updating the same Q in-place.

![mc-control1](https://user-images.githubusercontent.com/10358317/49828847-fbbd9a80-fd9d-11e8-9286-dee68c6fa1a2.jpg)

### MC Epsilon Greedy (without Exploring Starts) <a name="MCEpsilonGreedy"></a>:
- Demo Code: [monte_carlo_epsilon_greedy_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_epsilon_greedy_demo.ipynb)
- MC Exploring start is infeasible, because in real problems we can not calculate all edge cases (ex: in self-driving car problem, we can not calculate all cases).
- Randomly selection for all starting points in code is removed.
- Change policy to sometimes be random.
- This random policy is Epsilon-Greedy (like multi-armed bandit problem)

## Temporal Difference (TD) Learning Method <a name="TDLearning"></a>:
- Demo Code: [td0_prediction.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/td_prediction.ipynb)
- TD methods learn directly from episodes of experience.
- TD updates a guess towards a guess
- TD learns from incomplete episodes, by bootstrapping.
- TD uses bootstrapping like DP, TD learns experience like MC (combines MC and DP).

### MC - TD Difference <a name="MCTDDifference"></a>:
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

### MC - TD - DP Difference in Visual <a name="MCTDDifferenceinVisual"></a>:
![mc-td-dp](https://user-images.githubusercontent.com/10358317/49806522-01e55400-fd69-11e8-92a6-9bff14bb4c80.jpg)
[David Silver Lecture Notes]

### SARSA (TD Control Problem, On-Policy) <a name="SARSA"></a>:
- Demo Code: [SARSA_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/SARSA_demo.ipynb)
- In on-policy learning the Q(s,a) function is learned from actions, we took using our current policy π.

![updatingwithsarsa](https://user-images.githubusercontent.com/10358317/49831282-8c977480-fda4-11e8-8c7b-473ad5040f9d.jpg)
![sarsa-algo](https://user-images.githubusercontent.com/10358317/49831108-23affc80-fda4-11e8-84ca-08c6f1c056c5.jpg)
[David Silver Lecture Notes]


### Q-Learning (TD Control Problem, Off-Policy) <a name="Qlearning"></a>:
- Demo Code: [q_learning_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/q_learning_demo.ipynb)
- Looks like SARSA, instead of choosing a' based on argmax of Q, Q(s,a) is updated directly with max over  Q(s',a')
- In off-policy learning the Q(s,a) function is learned from different actions (for example, random actions). We even don't need a policy at all.

![qfunction](https://user-images.githubusercontent.com/10358317/49831386-d41e0080-fda4-11e8-967e-dd184a4e07e5.jpg)

![updatingwithqlearning](https://user-images.githubusercontent.com/10358317/49831118-26aaed00-fda4-11e8-9420-0ba120b1a509.jpg)
![qlearning-algo](https://user-images.githubusercontent.com/10358317/49831121-29a5dd80-fda4-11e8-9a72-aee5c9781950.jpg)
[David Silver Lecture Notes]

## Function Approximation <a name="FunctionApproximation"></a>:
- Demo Code: [func_approx_q_learning_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/func_approx_q_learning_demo.ipynb)
- Reinforcement learning can be used to solve large problems, e.g Backgammon:  10<sup>20</sup> states; Computer Go:  10<sup>170</sup> states; Helicopter: continuous state space.
- So far we have represented value function by a lookup table, Every state s has an entry V(s) or every state-action pair s,
a has an entry Q(s,a).
- There are too many states and/or actions to store in memory. It is too slow to learn the value of each state individually. Tabulated Q may not fit memory.
- Solution for large MDPs:

![func-appr](https://user-images.githubusercontent.com/10358317/49852371-fb042300-fdf4-11e8-8d15-1b77eb1c2fda.jpg)

- Differentiable function approximators can be used: Linear combinations of features, Neural Networks.

![value-func-appr](https://user-images.githubusercontent.com/10358317/49852598-a3b28280-fdf5-11e8-8a6d-c958136ca744.jpg)

![func-appr2](https://user-images.githubusercontent.com/10358317/49852876-66022980-fdf6-11e8-9820-380c39b280d0.jpg)
[David Silver Lecture Notes]

### Feature Vector <a name="FeatureVector"></a>:
![feature-vectors](https://user-images.githubusercontent.com/10358317/49852662-d2c8f400-fdf5-11e8-9550-5cf87360e964.jpg)
[David Silver Lecture Notes]

## Open AI Gym Environment <a name="OpenAIGym"></a>:
- Gym Framework is developed by OpenAI to simulate environments for RL problems (https://gym.openai.com/)
- Gym Q-learning Cart Pole implementation source code: https://github.com/omerbsezer/QLearning_CartPole
- Gym Q-learning Mountain Car implementation source code: https://github.com/omerbsezer/Qlearning_MountainCar

![gym](https://user-images.githubusercontent.com/10358317/50048143-b2d75000-00d5-11e9-85a5-93083ac9cd74.jpg)



## Policy-Based Methods <a name="PolicyBased"></a>:
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

### Policy Objective Functions <a name="PolicyObjectiveFunctions"></a>:
- Policy based reinforcement learning is an optimisation problem.
- Find θ that maximises J(θ).

![policy-objective-func](https://user-images.githubusercontent.com/10358317/49869176-a7a7ca00-fe20-11e8-8152-07a3ae3d00d7.jpg)
[David Silver Lecture Notes]

### Policy-Gradient <a name="PolicyGradient"></a>:
- Demo Code: [Pong_Game_Policy Gradient Implementation Using Gym and Tensorflow
](https://github.com/omerbsezer/PolicyGradient_PongGame)

![policy-gradient](https://user-images.githubusercontent.com/10358317/49869270-ffdecc00-fe20-11e8-9e2a-2811a3e1ecf0.jpg)

![score-function](https://user-images.githubusercontent.com/10358317/49869466-80053180-fe21-11e8-81b5-2cd4ef740609.jpg)

![policy-gradient-theorem](https://user-images.githubusercontent.com/10358317/49869560-cb1f4480-fe21-11e8-87eb-7ce93930038b.jpg)
[David Silver Lecture Notes]

### Monte-Carlo Policy Gradient (REINFORCE) <a name="REINFORCE"></a>:
![reinforce](https://user-images.githubusercontent.com/10358317/49869620-f1dd7b00-fe21-11e8-8023-a3de70e2cbbb.jpg)
[David Silver Lecture Notes]

## Actor-Critic <a name="ActorCritic"></a>:
- Actor-Critic method is a policy-based method (Learnt Value Function, Learnt Policy).

![actor-critique-intro](https://user-images.githubusercontent.com/10358317/49869844-b5f6e580-fe22-11e8-9a7d-974ea2147ba0.jpg)
[David Silver Lecture Notes]

- The critic is solving a familiar problem: policy evaluation.

### Action-Value Actor-Critic <a name="ActorValueActorCritic"></a>:
![action-value-actor-critic](https://user-images.githubusercontent.com/10358317/49869936-fe160800-fe22-11e8-8324-0f54c2d08680.jpg)

![advantage_func](https://user-images.githubusercontent.com/10358317/49873385-c744ef80-fe2c-11e8-9f09-f074b00f9ee3.jpg)
[David Silver Lecture Notes]

- The advantage function can significantly reduce variance of policy gradient.
- So the critic should really estimate the advantage function.

### Actor-critic algorithm: A3C <a name="A3C"></a>:
![a3c](https://user-images.githubusercontent.com/10358317/49952868-ec5d5f00-ff0d-11e8-83b1-e5bb4df41ae5.jpg)

### Different Policy Gradients <a name="DifferentPolicyGradients"></a>: 
![policy-gradient-summary](https://user-images.githubusercontent.com/10358317/49873726-a335de00-fe2d-11e8-9181-67b8c00e1bc3.jpg)
[David Silver Lecture Notes]

## Model-Based RL <a name="ModelBasedRL"></a>: 
![model-based-rl](https://user-images.githubusercontent.com/10358317/49874052-7e8e3600-fe2e-11e8-8c43-c554def10652.jpg)

![model-based-rl2](https://user-images.githubusercontent.com/10358317/49874157-ce6cfd00-fe2e-11e8-9551-b9d26fc760fb.jpg)
[David Silver Lecture Notes]

- Favourite planning algorithms: Value iteration,Policy iteration,Tree search,etc..
- Sample-based Modeling: A simple but powerful approach to planning. Use the model only to generate samples. Sample experience from model.
- Apply model-free RL to samples, e.g.: Monte-Carlo control, SARSA, Q-Learning.
- Model-based RL is only as good as the estimated model.
- When the model is inaccurate, planning process will compute a suboptimal policy: 1.when model is wrong, use model-free RL; 2.reason explicitly about model uncertainty.

### Real and Simulated Experience <a name="RealandSimulatedExperience"></a>: 
![real-simulated-exp](https://user-images.githubusercontent.com/10358317/49874473-a29e4700-fe2f-11e8-8685-e3baee5e626d.jpg)

![dyna-arch](https://user-images.githubusercontent.com/10358317/49874603-f90b8580-fe2f-11e8-9dc4-e85ec9caeaca.jpg)
[David Silver Lecture Notes]

### Dyna-Q Algorithm <a name="DynaQ"></a>: 
![dynaq](https://user-images.githubusercontent.com/10358317/49874998-fbbaaa80-fe30-11e8-8f0f-b266c22df8f8.jpg)
[David Silver Lecture Notes]

### Sim-Based Search <a name="SimBased"></a>: 
![sim-based-search](https://user-images.githubusercontent.com/10358317/49875256-9dda9280-fe31-11e8-8e5c-598f1929b843.jpg)
[David Silver Lecture Notes]

### MC-Tree-Search <a name="MCTreeSearch"></a>:
- AlphaGo- Supervised learning + policy gradients + value functions + Monte Carlo tree search D. Silver, A. Huang, C. J.Maddison, A. Guez, L. Sifre, et al. “Mastering the game of Go with deep neural networks and tree search”. Nature (2016).
- Highly selective best-first search.
- Evaluates states dynamically (unlike e.g.  DP).
- Uses sampling to break curse of dimensionality.
- Computationally efficient, anytime, parallelisable.
- Works for “black-box” models (only requires samples).

![mc-tree-search](https://user-images.githubusercontent.com/10358317/49875372-e6924b80-fe31-11e8-828b-58472a106e43.jpg)
[David Silver Lecture Notes]

### Temporal-Difference Search <a name="TemporalDifferenceSearch"></a>:
- Simulation-based search.
- Using TD instead of MC (bootstrapping).
- MC tree search applies MC control to sub-MDP from now.
- TD search applies Sarsa to sub-MDP from now.
- For simulation-based search, bootstrapping is also helpful.
- TD search is usually more efficient than MC search.
- TD(λ) search can be much more efficient than MC search.

![td-search](https://user-images.githubusercontent.com/10358317/49876106-a92ebd80-fe33-11e8-8459-bd38158edaad.jpg)
[David Silver Lecture Notes]

### RL in Games <a name="RLinGames"></a>:
![rl-in-games](https://user-images.githubusercontent.com/10358317/49876417-65888380-fe34-11e8-80d9-c50a84b3a1a2.jpg)
[David Silver Lecture Notes]

## Deep Q Learning (Deep Q-Networks: DQN) <a name="DQN"></a>:
- Gradient descent is simple and appealing. But it is not sample efficient.
- Batch methods seek to find the best fitting value function.
- Given the agent’s experience (“training data”)

### Experience Replay <a name="ExperienceReplay"></a>:
![dqn-experience-replay](https://user-images.githubusercontent.com/10358317/49853044-fa6c8c00-fdf6-11e8-94be-27a32fad21a2.jpg)

### DQN in Atari <a name="DQNAtari"></a>:
-V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, et al. “Playing Atari with Deep Reinforcement Learning”. (2013)

![dqn-in-atari](https://user-images.githubusercontent.com/10358317/49853048-fb9db900-fdf6-11e8-862d-d93acddceecf.jpg)

## Imitation Learning <a name="ImitationLearning"></a>:
- Video: https://www.youtube.com/watch?time_continue=1&v=WjFdD7PDGw0
- Presentation PDF: https://drive.google.com/file/d/12QdNmMll-bGlSWnm8pmD_TawuRN7xagX/view
- Given: demonstrations or demonstrator.
- Goal: train a policy to mimic demonstrations (mimicking human behavior).
- Pretraining model with human demonstrator’s data, it might avoid undesirable situations and make the training process faster.
- Behavior Cloning, Inverse RL, Learning from demonstration are the sub domain of the imitation learning.

![imitation-learning1](https://user-images.githubusercontent.com/10358317/49951502-22e5aa80-ff0b-11e8-9534-109e4499c265.jpg)

![imitation-learning2](https://user-images.githubusercontent.com/10358317/49951508-237e4100-ff0b-11e8-846f-5ff2c81b1fb7.jpg)

### Dagger: Dataset Aggregation <a name="Dagger"></a>:
Paper: https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf

![dagger](https://user-images.githubusercontent.com/10358317/49951510-24af6e00-ff0b-11e8-9ab6-06d949669a0e.jpg)

### PLATO: Policy Learning with Adaptive Trajectory Optimization <a name="PLATO"></a>:
- Kahn et al. ["PLATO: Policy Learning withAdaptive Trajectory Optimization"](https://arxiv.org/pdf/1603.00622.pdf) (2017)

![plato](https://user-images.githubusercontent.com/10358317/49951787-d2228180-ff0b-11e8-8c3e-386f57db782f.jpg)

### One-Shot Imitation Learning <a name="OneShotImitation"></a>:
- Video: https://www.youtube.com/watch?v=oMZwkIjZzCM

## Meta-Learning <a name="MetaLearning"></a>:
- Meta-learning = learning to learn
- Supervised meta-learning = supervised learning with datapoints that are entire datasets
- If we can meta-learn a faster reinforcement learner, we can learn new tasks efficiently!
- What can a  meta-learned learner do differently? 1.Explore more intelligently, 2.Avoid trying actions that are know to be useless, 3.Acquire the right features more quickly.
- The promise of meta-learning: use past experience to simply acquire a much more efficient deep RL algorithm

![meta-learning](https://user-images.githubusercontent.com/10358317/49952518-22e6aa00-ff0d-11e8-9008-43a459fae82d.jpg)


## POMDPs (Partial Observable MDP) <a name="POMDPs"></a>: 
![pomdps](https://user-images.githubusercontent.com/10358317/49738014-5c19e280-fc9f-11e8-8ca6-fe8fbeb0a5df.jpg)
[David Silver Lecture Notes]


## Resources <a name="Resources"></a>:

- [Deep Reinforcement Learning from Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/) , [Video](https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [GroundAI on RL](https://www.groundai.com/?text=reinforcement+learning): Papers on reinforcement learning
- [David Silver RL Lecture Notes](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Awesome RL - Github](https://github.com/aikorea/awesome-rl)
- [Free Deep RL Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
- [OpenAI - Spinning Up](http://spinningup.openai.com/)


## Important RL Papers <a name="ImportantRLPapers"></a>:
- Q-Learning: V.  Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, et al. [“Playing Atari with Deep Reinforcement Learning”](https://arxiv.org/pdf/1312.5602.pdf). (2013).
- V.  Mnih, K. Kavukcuoglu, D. Silver, et al. ["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (Nature-2015).
- Hasselt et al. ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/pdf/1710.02298.pdf) (2017).
- Hasselt et al. ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/pdf/1509.06461.pdf) (2015).
- Schaul et al. ["Prioritized Experience Replay"](https://arxiv.org/pdf/1511.05952.pdf) (2015).
- Wang et al. ["Dueling Network Architectures for Deep Reinforcement Learning"](http://proceedings.mlr.press/v48/wangf16.pdf) (2016).
- Fortunato et al. ["Noisy networks for exploration"](https://arxiv.org/pdf/1706.10295.pdf)(ICLR-2018).
- Sutton et al. ["Policy Gradient methods for reinforcement learning with function approximation"](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- Policy Gradient: V. Mnih et al, ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783.pdf) (2016).
- Policy Gradient: Schulman et al. ["Trust Region Policy Optimization"](https://arxiv.org/pdf/1502.05477.pdf) (2017).
- Schulma et al. ["Proximal Policy Optimization Algorithms"](https://arxiv.org/pdf/1707.06347.pdf)(2017).
- Such et al. ["Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for
Training Deep Neural Networks for Reinforcement Learning"](https://arxiv.org/pdf/1712.06567.pdf)(2018).
- Salimans et al. ["Evolution Strategies as a Scalable Alternative to Reinforcement Learning"](https://arxiv.org/pdf/1703.03864.pdf) (2017).
- Weber et al. ["Imagination-Augmented Agents for Deep Reinforcement Learning"](https://arxiv.org/pdf/1707.06203.pdf) (2018).
- Jaderberg et al. ["Reinforcement learning with unsupervised auxiliary tasks"](https://arxiv.org/pdf/1611.05397.pdf) (2016).
- Nagaband et al. ["Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"](https://arxiv.org/pdf/1708.02596.pdf)(2017)
- Robots-Guided policy search: S. Levine et al. [“End-to-end training of deep visuomotor policies”](https://arxiv.org/pdf/1504.00702.pdf). (2015).
- Robots-Q-Learning: D. Kalashnikov et al. [“QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation”](https://arxiv.org/pdf/1806.10293.pdf) (2018).
- AlphaGo- Supervised learning + policy gradients + value functions + Monte Carlo tree search D. Silver, A. Huang, C. J.Maddison, A. Guez, L. Sifre, et al. [“Mastering the game of Go with deep neural networks and tree search”](https://www.nature.com/articles/nature16961). Nature (2016).
- Kahn et al. ["PLATO: Policy Learning withAdaptive Trajectory Optimization"](https://arxiv.org/pdf/1603.00622.pdf) (2017)


## References <a name="References"></a>:
- [Sutton & Barto Book: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
- David Silver RL Lecture Notes: (http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Imitation Learning: https://sites.google.com/view/icml2018-imitation-learning/
- Udemy: https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/ 
- Udemy: https://www.udemy.com/deep-reinforcement-learning-in-python/ 
- Meta Learning: http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_16_meta_learning.pdf

