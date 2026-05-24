---
tags:
  - ReinforcementLearning
---
## Basics About Reinforcement Learning 
- RL used to solve a **controlled tasks**
- **Elements common to all control tasks**
	- State
	- Action
	- Reward
	- Return
	- Agent
	- Environment
- **Agent:** The learner/algo
- **Environment:** The world the agent interacts with
	- Types:
		- Fully observable environment: where state == observation, like chess game
		- Fully observable environment: where state == observation, like self driving car
- **Model:**
	- It simulate the environment and give prediction to the ai agent
	- Most of the time we do not have a model
	- ![[Pasted image 20260329123931.png]]
	- Types of RL Problem on the basis of model availability:
		- Model-based Reinforcement Problem: Problem where we have the model
		- Model-based Reinforcement Problem: Problem where we did not have the model
- **Episode vs Continuing Task and Trajectory:**
	- Episode: a sequence with a clear end (like one game round).
	- Continuing: a never-ending task. (like Autonomous Traffic Light Control System)
	- Trajectory: A trajectory $\tau$ is the full sequence of states, actions, and rewards
		- $\tau = (s_0, a_0, r_0,\ s_1, a_1, r_1,\ s_2, a_2, r_2,\ \dots,\ s_T, a_T, r_T)$
- **State:**
	- A description of the situation the agent finds itself in.
	- Denoted by ($S_t$) which mean State (S) at timestamp (t)
- **Action:** 
	- A choice the agent make at a moment.
	- Denoted by ($A_t$) which mean Action (A)  at timestamp (t) 
- **Policy:** 
	- Thing of it as a function which take state and return action. 
	- Denoted by ($π_t$) which mean Policy ($π$) at timestamp (t)
	- $π(a|s)$ = Probability of taking action 'a' in a state s
	- $π(s)$ = Action 'a' taken in a state 's'
- **Reward:**
	- Immediate feedback after one action
	- Note: the agent does not want to maximize the Reward
	- Denoted by ($R_t$) which mean Reward (R) at timestamp (t)
- **Return:**
	- Also know as Cumulative Reward, Denoted by ($G_t$)
	- Actual total future reward from timestamp t onward (often discounted)
	- Can be calculated for each time step.
	- Agent goal is to maximise this
	- Formula
		- Summation form:
			- $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$
		- Expanded (unrolled) form:
			- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots$
		- Recursive form:
			- $G_t = R_{t+1} + \gamma G_{t+1}$
		- here
			- t= timestamp for which we want to calculate the Return 
			- k= it will keep increasing
			- $\gamma$  is discount factor
	- What is **Discount Factor**:
		- Discount Factor shrinks the future rewards
		- 0 ≤ $\gamma$ ≤ 1 Lower $\gamma$ → short-sighted; $\gamma$ close to 1 → long-term planning.
		- **Key insight:** Future rewards are multiplied by $\gamma^t$, so sooner rewards are always worth more than later ones. This makes the agent prefer:
			- Present over future (time preference)
			- Faster solutions over slower ones (efficiency)

| Time step | Reward | y=0 (fully  future blind) | y=0.3 (very short-sighted) | y=0.9 (long term) | y=1 (no discount) |
| --------- | ------ | ------------------------- | -------------------------- | ----------------- | ----------------- |
| t=0       | 1      | 1                         | 1                          | 1                 | 1                 |
| t=1       | 1      | 0                         | 0.3                        | 0.9               | 1                 |
| t=2       | 1      | 0                         | 0.09                       | 0.81              | 1                 |
| t=3       | 1      | 0                         | 0.027                      | 0.729             | 1                 |
| t=4       | 1      | 0                         | 0.008                      | 0.656             | 1                 |
| t=5       | 1      | 0                         | 0.002                      | 0.59              | 1                 |
| Total G   |        | 1                         | 1.428                      | 4.686             | 6                 |

- **Markov decision processes**
	- A Markov Decision Process is a setup where an agent makes decisions step-by-step, the future depends only on the current situation and action.
	- ![[Pasted image 20260329144239.png]]
	- **Types**
		- **Infinite:** One or more these 3 values(state , action and rewards) are infinite for a controlled task
		- **Finite:** Finite number of states, action and rewards for a controlled task
	- Describing a controlled task with the help of markov decision process
		- **(S, A, R, P)**
		- S= Set of possible state of the task
		- A= Set of action that can be taken in each of the state
		- R= Set of  rewards for each (s,a) pair
		- P= Probability of passing from one state to another when taking each possible action
- **Value function**
	- **State Value:** The value of being in state s = expected return 'G(t)' from state 's'
		- $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s] = \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$
			- $v_\pi(s)$ = State value
			- $\mathbb{E[]}$ = Expected Maths symbol [[Expected]]
			- $\mathbb{E}[G_t \mid S_t = s]$ = Expected G(t) when S(t) = s{given that right now I'm in state 's'}
	- **Action Value/Q Value:** Expected return from state 's' and taking action 'a'
		- $q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a] = \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]$
		- $q_\pi(s,a)$ = Action value
- **Bellman Equations:**
	- **Bellman equations for state value:**
		 - $\mathbb{E}[G_{t+1} \mid S_{t+1} = s'] = v_\pi(s')$ by definition. So: $\boxed{v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma v_\pi(s')\Big]}$
	- **Bellman equations for action value:**
		- $\boxed{q_\pi(s,a) = \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a')\right]}$
- **Exploration vs Exploitation**
	- Should the agent try new actions to learn (explore) or use its current best action to get reward (exploit)?
	- Analogy: Trying a new route in a race (maybe faster) vs sticking to the route that already works.
	- Tech note: Balancing these is central to RL (epsilon-greedy, UCB, etc.).
## Algo
- Tabular methods
	- Dynamic Programming
	- Monte carlo
	- Temporal Difference
	- N-Step bootstrapping
	- State aggregation
	- Tile coding
- Artificial Neural Networks
	- Deep Sarsa
	- Deep Q-Learning
- Policy Gradient Methods
	- Reinforce A2C
