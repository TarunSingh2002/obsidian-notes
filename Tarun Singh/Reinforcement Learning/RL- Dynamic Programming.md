---
tags:
  - ReinforcementLearning
---
## Bellman Equation (from zero) + Dynamic Programming

- **Roadmap of this note**
	- Part 1: Where the Bellman equation comes from (slow derivation, starting from Return)
	- Part 2: What every symbol means
	- Part 3: Applying it when we **know everything** (hand example)
	- Part 4: Bellman Optimality equation (average → max)
	- Part 5: When we **don't know the values** → Dynamic Programming (Policy Evaluation, Value Iteration, Policy Iteration) with hand example
	- Part 6: Python code
	- Part 7: Cheat sheet

---

## **Part 1: Derivation of the Bellman Equation (slow)**

- **Step 0 — What we already know (from Basics note)**
	- Return: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$
	- Recursive form: $G_t = R_{t+1} + \gamma G_{t+1}$
	- Why recursive form is true (just factor out one $\gamma$):
		- $G_t = R_{t+1} + \gamma \underbrace{\left(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots\right)}_{\text{this bracket is exactly } G_{t+1}}$
		- In words: **Total from now = reward I get next + γ × total from the next step onward**

- **Step 1 — The problem: $G_t$ is random**
	- If I stand in the same state twice, I can get a different $G_t$ each time. Why?
		- **Reason 1 (policy is random):** policy $\pi(a|s)$ may pick different actions (e.g. 50% left, 50% right)
		- **Reason 2 (environment is random):** same action can lead to different next states/rewards (e.g. slippery floor)
	- So one single $G_t$ doesn't tell me "how good is this state". I need the **average** $G_t$.
	- That average is called the **State Value**:
		- $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$
		- Read it as: "If I start in state $s$ and follow policy $\pi$, what Return do I get **on average**?"

- **Step 2 — Quick reminder: what is Expected value $\mathbb{E}$ ?**
	- Expected = **sum of (probability × value)** over every possible outcome
	- Example: a game where 50% chance you get ₹10 and 50% chance you get ₹2
		- $\mathbb{E} = 0.5 \times 10 + 0.5 \times 2 = 6$
	- That's it. Every $\mathbb{E}$ below just means "list all possibilities, multiply each by its probability, add them up".

- **Step 3 — Put the recursive Return inside the value**
	- $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$
	- Replace $G_t$ with $R_{t+1} + \gamma G_{t+1}$:
	- $v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$
	- Nothing new here, just substitution.

- **Step 4 — List all the possibilities ("the tree")**
	- From state $s$, what can happen in **one step**?
		- First the **agent** picks action $a$ → probability $\pi(a|s)$
		- Then the **environment** gives next state $s'$ and reward $r$ → probability $p(s', r \mid s, a)$
	- So the probability of one full path "pick $a$ → land in $s'$ with reward $r$" is:
		- $\pi(a|s) \times p(s', r \mid s, a)$
	- Picture:
```
                 s            ← I am here
               /   \
       π(a1|s)/     \π(a2|s)  ← agent chooses (policy)
             a1      a2
            / \      / \
      p(..) /   \    /   \     ← environment decides (model / dynamics)
         s',r  s',r s',r  s',r ← where I land + reward I got
```

- **Step 5 — Expected = sum over the tree (probability × value)**
	- Using Step 2's rule ("sum of probability × value"):
	- $v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \times \Big[\text{what } R_{t+1} + \gamma G_{t+1} \text{ is on that branch}\Big]$
	- On a branch, $R_{t+1}$ is just the reward $r$ of that branch. Easy.
	- But $G_{t+1}$ is still random (future after $s'$ is still unknown). What do we put for it?

- **Step 6 — The magic step: average of $G_{t+1}$ is $v_\pi(s')$**
	- Once we are in $s'$, the average Return from there is, **by definition**:
		- $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] = v_\pi(s')$
	- Why can we ignore how we reached $s'$? → **Markov property**: the future depends only on the current state, not on the past.
	- So on each branch, replace $G_{t+1}$ with $v_\pi(s')$.

- **Step 7 — Final Result: Bellman Expectation Equation (for state value)**
	- $\boxed{v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_\pi(s')\Big]}$
	- In plain words: **Value of here = average over everything that can happen of (reward now + γ × value of where I land)**
	- The beautiful part: the infinite future ($G_{t+1}$ = infinite sum) got replaced by **one number** $v_\pi(s')$. So we only ever look **one step ahead**.

- **Same idea for Action Value (Q value)**
	- $q_\pi(s,a)$ = average Return if I'm in $s$, take action $a$ **now** (fixed), then follow $\pi$
	- Action is already fixed, so we skip the $\pi(a|s)$ layer:
		- $q_\pi(s,a) = \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_\pi(s')\Big]$
	- And state value is just average of Q values over the policy:
		- $v_\pi(s) = \sum_a \pi(a|s)\, q_\pi(s,a)$
	- (Plug second into first's $v_\pi(s')$ and you get the Q-version in the Basics note.)

---

## **Part 2: What Each Part Means**

| Part | Meaning | Who controls it |
|---|---|---|
| $s$ | Current state (where I am) | - |
| $a$ | An action I could take | - |
| $s'$ | Next state (where I land) | - |
| $r$ | Reward I get on this step | Environment |
| $v_\pi(s)$ | Average Return from $s$ if I follow $\pi$ | - |
| $v_\pi(s')$ | Average Return from the next state (the "future" squeezed into one number) | - |
| $\pi(a \mid s)$ | Probability I choose action $a$ in $s$ | **Agent** (policy) |
| $p(s', r \mid s, a)$ | Probability environment sends me to $s'$ with reward $r$ after $a$ | **Environment** (model) |
| $\sum_a$ | "Go over every action" | - |
| $\sum_{s',r}$ | "Go over every possible result of that action" | - |
| $\gamma$ | Discount: future is worth a bit less | Us (design choice) |
| $r + \gamma v_\pi(s')$ | Value of **one branch** = reward now + discounted future | - |

- **Intuition:** "How good is my position?" = "average of (what I get right now + how good my next position is)"
- **Deterministic special case** (only one action, and it always goes to the same $s'$): all probabilities are 1, so
	- $v(s) = r + \gamma\, v(s')$ ← exactly the recursive Return formula, but with values. That's the connection with $G_t = R_{t+1} + \gamma G_{t+1}$.

---

## **Part 3: Applying Bellman when we KNOW everything (hand example)**

- **Setup**
	- We are in state **S**. Policy: 50% Left, 50% Right → $\pi(L|S)=0.5,\ \pi(R|S)=0.5$
	- $\gamma = 0.9$
	- Values of next states are **already known** ("we know the future"): $v(S_1)=10,\ v(S_2)=5,\ v(S_3)=0$
	- What the environment does:

| Action | Lands in | Probability $p$ | Reward $r$ | $v(s')$ |
|---|---|---|---|---|
| Left | $S_1$ | 1.0 | +2 | 10 |
| Right | $S_2$ | 0.8 | 0 | 5 |
| Right | $S_3$ | 0.2 | +10 | 0 |

```
                  S
          0.5 /       \ 0.5
           Left       Right
            |  1.0    0.8 /  \ 0.2
           S1(r=2)  S2(r=0)  S3(r=10)
           v=10      v=5      v=0
```

- **Step 1 — Value of each branch: $r + \gamma v(s')$**
	- Left → S1: $2 + 0.9 \times 10 = 2 + 9 = 11$
	- Right → S2: $0 + 0.9 \times 5 = 4.5$
	- Right → S3: $10 + 0.9 \times 0 = 10$

- **Step 2 — Average inside each action (environment layer, use $p$) = Q values**
	- $q(S, L) = 1.0 \times 11 = 11$
	- $q(S, R) = 0.8 \times 4.5 + 0.2 \times 10 = 3.6 + 2 = 5.6$

- **Step 3 — Average over actions (policy layer, use $\pi$)**
	- $v_\pi(S) = 0.5 \times 11 + 0.5 \times 5.6 = 5.5 + 2.8 = \mathbf{8.3}$

- **Takeaway:** always work **bottom-up**: branch value → average with $p$ → average with $\pi$.
- **Notice:** Left (11) is way better than Right (5.6). The random policy is wasting value. That leads to the optimality equation 👇

---

## **Part 4: Bellman Optimality Equation (average → max)**

- **Expectation equation** = "how good is state $s$ for **this given policy**" → **average** over actions
- **Optimality equation** = "how good is $s$ if I act **perfectly**" → pick the **best** action, i.e. **max**
	- $\boxed{v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_*(s')\Big]}$
	- $\boxed{q_*(s,a) = \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma \max_{a'} q_*(s', a')\Big]}$
- Only change: $\sum_a \pi(a|s)$ (average) → $\max_a$ (best one)
- Note: we still average over the **environment** ($p$) because we can't control the environment, only our own choice.
- **In Part 3's example:** $v_*(S) = \max(11, 5.6) = \mathbf{11}$ → best action = **Left**
- **Optimal policy from values:** $\pi_*(s) = \arg\max_a q_*(s,a)$ → "pick the action with the biggest Q"

---

## **Part 5: When we DON'T know the values → Dynamic Programming**

- **The problem**
	- In Part 3, $v(S_1), v(S_2), v(S_3)$ were given. In real life they are **unknown** — they're exactly what we want to find!
	- Bellman says $v(A)$ depends on $v(B)$, and $v(B)$ depends on $v(A)$... chicken-and-egg.
- **The DP trick: guess, then repeat**
	- Start with a guess: all values = 0
	- Use Bellman equation to compute new values **from the old guesses**
	- Repeat. Every round the values get more correct (info from the real rewards spreads backward one step per round)
	- Stop when values stop changing (change < small θ like 0.0001)
- **What DP needs:** the **model** $p(s', r \mid s, a)$ must be known → DP works for **model-based** problems only (like Gym's FrozenLake where we can read `env.P`)

- **Our tiny world (corridor)**
```
  [ A ] — [ B ] — [ G ]
                  goal (terminal, v(G) = 0 always)
```
	- States: A, B, G (G = goal, episode ends)
	- Actions: Left, Right
	- Moves are deterministic. Walking Left from A hits the wall → stay in A.
	- Reward: **−1 for every step** (so the agent wants to reach G fast)
	- $\gamma = 0.9$

| State | Action | Next state | Reward |
|---|---|---|---|
| A | Left | A (wall) | −1 |
| A | Right | B | −1 |
| B | Left | A | −1 |
| B | Right | G | −1 |

### **5.1 Policy Evaluation ("how good is my current policy?")**
- **Goal:** find $v_\pi$ for a **fixed** policy. Use the random policy: 50% Left, 50% Right
- **Update rule** (Bellman expectation equation used as an update):
	- $V_{new}(s) = \sum_a \pi(a|s) \big[r + \gamma V_{old}(s')\big]$  (deterministic, so $p = 1$)
- **Write it for our states**
	- $V(A) = 0.5\,[-1 + 0.9\,V(A)] + 0.5\,[-1 + 0.9\,V(B)]$
	- $V(B) = 0.5\,[-1 + 0.9\,V(A)] + 0.5\,[-1 + 0.9 \times 0]$ (right goes to G, and $V(G)=0$)
- **Iteration 0:** $V(A)=0,\ V(B)=0$
- **Iteration 1** (use old values 0, 0)
	- $V(A) = 0.5[-1 + 0] + 0.5[-1 + 0] = -1$
	- $V(B) = 0.5[-1 + 0] + 0.5[-1] = -1$
- **Iteration 2** (old: A=−1, B=−1)
	- $V(A) = 0.5[-1 + 0.9(-1)] + 0.5[-1 + 0.9(-1)] = 0.5(-1.9) + 0.5(-1.9) = -1.9$
	- $V(B) = 0.5[-1 + 0.9(-1)] + 0.5[-1] = -0.95 - 0.5 = -1.45$
- **Iteration 3** (old: A=−1.9, B=−1.45)
	- $V(A) = 0.5[-1 + 0.9(-1.9)] + 0.5[-1 + 0.9(-1.45)] = 0.5(-2.71) + 0.5(-2.305) = -1.355 - 1.1525 = -2.5075$
	- $V(B) = 0.5[-1 + 0.9(-1.9)] + 0.5[-1] = -1.355 - 0.5 = -1.855$
- **Keep going...**

| Iteration | V(A) | V(B) | Biggest change |
|---|---|---|---|
| 0 | 0 | 0 | - |
| 1 | −1.000 | −1.000 | 1.000 |
| 2 | −1.900 | −1.450 | 0.900 |
| 3 | −2.508 | −1.855 | 0.608 |
| 4 | −2.963 | −2.128 | 0.456 |
| 5 | −3.291 | −2.333 | 0.328 |
| 10 | −3.992 | −2.766 | 0.067 |
| 20 | −4.165 | −2.873 | 0.003 |
| 35 | −4.173 | −2.878 | ~0 ✅ converged |

- **Final:** $v_\pi(A) \approx -4.17,\ v_\pi(B) \approx -2.88$
- **Meaning:** with random moves, from A you waste a lot of steps (bumping wall, going back) → very negative. From B you're closer to G → less negative.
- **Check it's really correct:** plug into Bellman: $V(B) = 0.5[-1 + 0.9(-4.17)] + 0.5[-1] = -2.38 - 0.5 = -2.88$ ✅ (left side = right side, that's what "solution of Bellman equation" means)

### **5.2 Policy Improvement ("can I do better?")**
- Use the values we just found, compute Q for each action, pick the best (greedy):
	- A, Left: $-1 + 0.9(-4.17) = -4.75$
	- A, Right: $-1 + 0.9(-2.88) = -3.59$ ← better
	- B, Left: $-1 + 0.9(-4.17) = -4.75$
	- B, Right: $-1 + 0.9(0) = -1$ ← better
- **New policy:** A → Right, B → Right

### **5.3 Policy Iteration (= Evaluation + Improvement in a loop)**
- Loop:
	1. **Evaluate** current policy (5.1)
	2. **Improve** it greedily (5.2)
	3. If policy didn't change → stop, it's optimal
- In our corridor:
	- Round 1: random policy → values (−4.17, −2.88) → new policy (Right, Right)
	- Round 2: evaluate (Right, Right): $V(B) = -1 + 0.9(0) = -1$, $V(A) = -1 + 0.9(-1) = -1.9$ → improve → still (Right, Right) → **stop** ✅

### **5.4 Value Iteration (shortcut: use max directly)**
- Instead of fully evaluating then improving, just use the **optimality equation** as the update:
	- $V_{new}(s) = \max_a \big[r + \gamma V_{old}(s')\big]$
- **Iteration 0:** $V(A)=0,\ V(B)=0$
- **Iteration 1** (old 0, 0)
	- $V(A) = \max(\underbrace{-1 + 0.9(0)}_{Left},\ \underbrace{-1 + 0.9(0)}_{Right}) = \max(-1, -1) = -1$
	- $V(B) = \max(\underbrace{-1 + 0.9(0)}_{Left},\ \underbrace{-1 + 0}_{Right \to G}) = -1$
- **Iteration 2** (old: A=−1, B=−1)
	- $V(A) = \max(-1 + 0.9(-1),\ -1 + 0.9(-1)) = \max(-1.9, -1.9) = -1.9$
	- $V(B) = \max(-1 + 0.9(-1),\ -1 + 0) = \max(-1.9, -1) = -1$
- **Iteration 3** (old: A=−1.9, B=−1)
	- $V(A) = \max(-1 + 0.9(-1.9),\ -1 + 0.9(-1)) = \max(-2.71, -1.9) = -1.9$
	- $V(B) = \max(-1 + 0.9(-1.9),\ -1) = \max(-2.71, -1) = -1$
	- Nothing changed → **converged** ✅

| Iteration | V(A) | V(B) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | −1 | −1 |
| 2 | −1.9 | −1 |
| 3 | −1.9 | −1 ✅ |

- **Read off the policy** (argmax in last iteration): A → Right (−1.9 > −2.71), B → Right (−1 > −2.71)
- **Sanity check:** from A, best path = A→B→G = 2 steps: $-1 + 0.9 \times (-1) = -1.9$ ✅ matches!
- Same answer as Policy Iteration, but in 3 tiny rounds.

### **5.5 Policy Evaluation vs Value Iteration**

| | Policy Evaluation | Value Iteration |
|---|---|---|
| Question it answers | How good is **this** policy? | What is the **best** possible value? |
| Bellman equation used | Expectation (average over $\pi$) | Optimality (max over actions) |
| Update | $\sum_a \pi(a\|s)[r + \gamma V(s')]$ | $\max_a [r + \gamma V(s')]$ |
| Gives policy? | No (need improvement step) | Yes (argmax at the end) |
| Our example result | A = −4.17, B = −2.88 | A = −1.9, B = −1 |

- Note: some code updates $V$ **in place** (uses the new value of A immediately when computing B). It converges a bit faster but gives the same final answer. Above I used old values for clarity.

---

## **Part 6: Python Code (same corridor)**

```python
gamma = 0.9
theta = 1e-4                       # stop when change is smaller than this

# model: P[state][action] = list of (probability, next_state, reward)
P = {
    'A': {'L': [(1.0, 'A', -1)], 'R': [(1.0, 'B', -1)]},
    'B': {'L': [(1.0, 'A', -1)], 'R': [(1.0, 'G', -1)]},
}
terminal = {'G'}

def q_value(V, s, a):
    # sum over environment outcomes: p * (r + gamma * V(s'))
    return sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a])

# ---------- Policy Evaluation (random policy) ----------
policy = {'A': {'L': 0.5, 'R': 0.5}, 'B': {'L': 0.5, 'R': 0.5}}
V = {'A': 0.0, 'B': 0.0, 'G': 0.0}           # step 0: guess all zeros
while True:
    delta = 0
    newV = dict(V)
    for s in P:                               # every non-terminal state
        # Bellman expectation: average over actions using pi(a|s)
        newV[s] = sum(policy[s][a] * q_value(V, s, a) for a in P[s])
        delta = max(delta, abs(newV[s] - V[s]))
    V = newV
    if delta < theta:                         # values stopped changing
        break
print("Policy evaluation:", V)    # A ≈ -4.17, B ≈ -2.88

# ---------- Value Iteration ----------
V = {'A': 0.0, 'B': 0.0, 'G': 0.0}
while True:
    delta = 0
    newV = dict(V)
    for s in P:
        # Bellman optimality: take the BEST action (max) instead of average
        newV[s] = max(q_value(V, s, a) for a in P[s])
        delta = max(delta, abs(newV[s] - V[s]))
    V = newV
    if delta < theta:
        break
best_policy = {s: max(P[s], key=lambda a: q_value(V, s, a)) for s in P}
print("Value iteration:", V)      # A = -1.9, B = -1.0
print("Best policy:", best_policy) # {'A': 'R', 'B': 'R'}
```

---

## **Part 7: Cheat Sheet**

- $G_t = R_{t+1} + \gamma G_{t+1}$ → take average on both sides → $v(s) = \text{avg}[\,r + \gamma v(s')\,]$
- **Two layers of averaging:** agent's choice ($\pi$) and environment's result ($p$)
- **Expectation eq.** → average over actions → evaluate a policy
- **Optimality eq.** → max over actions → find the best policy
- **Know all values?** → plug in once, bottom-up (branch → $p$ → $\pi$)
- **Don't know values?** → DP: start at 0, apply Bellman again and again until values stop changing
- **DP needs the model** $p(s',r|s,a)$. No model → later methods (Monte Carlo, TD, Q-learning) learn from experience instead
- **Remember the shape, not the symbols:** "value here = (reward now) + γ × (value of where I land), averaged (or maxed) over what can happen"
