---
tags:
  - ReinforcementLearning
  - dynamic-programming
---
## Bellman Equation + Dynamic Programming (apply-first version)

---

## Basics of DP
- We can only apply Dp to the projects where we already know the value of the $p(s', r \mid s, a)$ , For example board games, simulators or Gym’s FrozenLake, When p is unknown or keeps changing, you can’t use DP. That’s exactly why Monte Carlo, TD and Q-learning exist, the next topics in your course. They learn from playing instead of from p.
- The best policy is fixed, so is that memorization? -> Yes, it is, because why gamble on a worse move? It isn’t memorization: it’s a rule for _every_ state, computed from the rules. If the maze changes, p changes, so you re-run value iteration and get a new policy. Randomness (epsilon-greedy) matters later, when you don’t know the rules and have to explore.
- *Math equation we used here (Bellman Equations)*
	- **Bellman equations:*
		 - $$\boxed{v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_\pi(s')\Big]}$$
		- $$q_\pi(s,a) = \sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, v_\pi(s')\big]$$
	- **Bellman optimality equations:**
		- $$\boxed{v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_*(s')\Big]  = \max_a q(s,a)}$$
		- $$q(s,a) = \sum_{s',r} p\,[r + \gamma v(s')]$$
- In real problems nobody gives us $v(s')$. And $v(A)$ needs $v(B)$, while $v(B)$ needs $v(A)$... chicken and egg 🐔🥚
- **DP trick: guess, then repeat**
	1. Start with a guess: **all values = 0**
	2. For every state, plug the **old guesses** into the right side of the Bellman equation → get **new values**
	3. Repeat step 2 again and again
	4. Stop when the values stop changing (change smaller than a tiny number θ, like 0.0001)
- **The 3 DP algorithms:**

| Algorithm         | What it does                                                           | Equation used                    |
| ----------------- | ---------------------------------------------------------------------- | -------------------------------- |
| Policy Evaluation | Finds hidden $v_\pi$ for a **given** policy                            | Expectation (average with $\pi$) |
| Policy Iteration  | Evaluate → improve policy → evaluate → ... until policy stops changing | Expectation + "pick best q"      |
| Value Iteration   | Directly finds the best values and best policy                         | Optimality (max)                 |




## Part 0: Math we should know

- **Conditional probability**
	- $P(A \mid B)$ = probability of A **given** B has already happened. Read "|" as "given".
	- Formula: $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$
	- Example: a card is drawn from a deck. Probability it is a King **given** it is a face card?
		- Total cards = 52, face cards = 12
		- A = "card is a King", B = "card is a face card"
		- $P(B) = \frac{12}{52}$, $P(A \cap B) = \frac{4}{52}$
		- $P(A \mid B) = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}$

- **Chance of "A and then B"**
	- If B does **not** depend on A: chance = $P(A) \times P(B)$
		- Coin then dice: chance of "Heads and then a 6" = $\frac{1}{2} \times \frac{1}{6} = \frac{1}{12}$
	- If B **depends** on A: chance = $P(A) \times P(B \mid A)$
		- Bag has 3 red, 2 blue balls. Pick 2 without putting back. Chance both red = $\frac{3}{5} \times \frac{2}{4} = 0.3$

- **Expected value $\mathbb{E}$**
	- Expected value = **the average result you get if you repeat the same thing many, many times**
	- **Rule (always the same 3 moves):** list every possible outcome → multiply each by its probability → add
		- $\mathbb{E}[X] = \sum_{\text{all outcomes } x} P(x) \times x$
		- All the probabilities must add up to 1
	- **Example 1 — Simple game:** 50% you get ₹10, 50% you get ₹2
		- $\mathbb{E} = 0.5 \times 10 + 0.5 \times 2 = 6$ → play 100 times, you get about ₹600, so ₹6 per game
	- **Example 2 — Dice:** each face has chance $\frac{1}{6}$
		- $\mathbb{E} = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5$ → you can never roll 3.5, an average doesn't need to be a real outcome
	- **Example 3 — Unequal chances:** 20% win ₹50, 80% lose ₹10
		- $\mathbb{E} = 0.2 \times 50 + 0.8 \times (-10) = 10 - 8 = 2$
	- **Example 4 — RL flavour:** I press "jump". 70% I land safely and get +5, 30% I fall and get −10
		- $\mathbb{E}[\text{reward}] = 0.7 \times 5 + 0.3 \times (-10) = 3.5 - 3 = 0.5$
		- 👉 Remember this one. Later, **q value is exactly this kind of calculation.**

---

## Part 1: The words / symbols

| Symbol               | Meaning (simple)                                                                         | Who decides it     |
| -------------------- | ---------------------------------------------------------------------------------------- | ------------------ |
| $s$                  | Current state (where I am)                                                               | -                  |
| $a$                  | An action I can take                                                                     | Agent              |
| $s'$                 | Next state (where I land after the action)                                               | Environment        |
| $r$                  | Reward I get on this step (a normal number like +1, −1, +10)                             | Environment        |
| $\gamma$             | Discount factor (0 to 1): future reward is worth a bit less                              | Us (design choice) |
| $\pi(a \mid s)$      | Policy: probability that I choose action $a$ in state $s$                                | **Agent**          |
| $p(s', r \mid s, a)$ | Model: probability I land in $s'$ and get reward $r$, **given** I was in $s$ and did $a$ | **Environment**    |
| $v_\pi(s)$           | State value: average total reward (Return) from $s$ if I follow $\pi$                    | Computed           |
| $q_\pi(s,a)$         | Action value (Q value): average Return if I do $a$ in $s$, then follow $\pi$             | Computed           |
| $\sum_a$             | "for every action" (a for loop)                                                          | -                  |
| $\sum_{s',r}$        | "for every result that can happen after this action" (a for loop)                        | -                  |

- **3 things people get confused about (I did too):**
	- **$s'$ is not one fixed state.** It's a blank like $x$ = "**whichever** state I land in". If an action can land me in 2 places, $s'$ takes 2 values, one at a time. Same for $r$.
	- **$r$ is not between 0 and 1.** $p$ is between 0 and 1 (it's a probability). $r$ is the reward the game gives, like +1 or −10.
	- **How many terms in $\sum_{s',r}$?** Only the results that can **actually happen** after that action. Every other state has $p = 0$, and $0 \times \text{anything} = 0$, so we skip it. Usually it's 1 to 4 terms.

| Example | How many results after one action |
| --- | --- |
| Corridor (Part 5), not slippery | 1 (you always go where you want) |
| FrozenLake slippery (Gym) | 3 (forward, slip left, slip right) |
| Robot on ice (Part 3) | Left: 1, Right: 2 |

---

## Part 2: THE equation (this is what we work with)

- **Bellman Expectation Equation** ("how good is each state if I follow policy $\pi$")
	- $\boxed{v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_\pi(s')\Big]}$
- **It is really 2 small equations stacked together.** Read it from inside to outside:

| Piece | Plain words | Name |
| --- | --- | --- |
| $r + \gamma\, v_\pi(s')$ | Total for one result = reward now + γ × how good the place I land is | - |
| $\sum_{s',r} p(s', r \mid s, a) \times [\dots]$ | Average over all results of **one action** (use the environment's chances $p$) | $q_\pi(s,a)$ |
| $\sum_a \pi(a \mid s) \times [\dots]$ | Average over all **actions** (use my policy's chances $\pi$) | $v_\pi(s)$ |

- So the same equation written in 2 steps:
	- **Step A (one action):** $q_\pi(s,a) = \sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, v_\pi(s')\big]$
	- **Step B (all actions):** $v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s,a)$
- **Bellman Optimality Equation** ("how good is each state if I act **perfectly**")
	- $\boxed{v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\Big[r + \gamma\, v_*(s')\Big]}$
	- Only change: Step B uses **max** (pick the best action) instead of the average with $\pi$
	- Step A stays the same, because we can't control the environment, so we still average over $p$
	- In 2 steps: $q(s,a) = \sum_{s',r} p\,[r + \gamma v(s')]$, then $v_*(s) = \max_a q(s,a)$

- **⭐ What is known and what is hidden?** (very important)

| Thing                      | Known or hidden? | Where it comes from              |
| -------------------------- | ---------------- | -------------------------------- |
| States, actions            | Known            | The problem/game                 |
| $r$ (rewards)              | Known            | The game rules                   |
| $p(s', r \mid s, a)$       | Known **in DP**  | The game rules (in Gym: `env.P`) |
| $\gamma$                   | Known            | We choose it                     |
| $\pi(a \mid s)$            | Known            | We choose it (e.g. 50/50 random) |
| $v_\pi(s)$ for every state | **Hidden** ❗     | This is what we want to find     |

- Notice: $v$ appears on **both sides** of the equation. To find $v(s)$ I need $v(s')$, which is also hidden.
	- In **Part 3** I'll pretend someone **tells us** the $v(s')$ values, just to practise plugging in. In real problems they are **hidden**.
	- In **Part 4–7** we find those hidden values with DP.

---

## Part 3: Applying the equation when values are KNOWN (robot on ice)

- **Setup**
	- I'm a robot on ice in state **Start**. $\gamma = 0.9$
	- My policy: $\pi(\text{Left} \mid \text{Start}) = 0.5$, $\pi(\text{Right} \mid \text{Start}) = 0.5$
	- Values of next states: $v(\text{Safe}) = 10$, $v(\text{Gold}) = 20$, $v(\text{Hole}) = 0$
		- ⚠️ These are normally **hidden**. Here we pretend someone gave them to us, just to practise. (In Part 5 you'll see how DP finds such numbers.)
	- What can happen (game rules = known):

| Action | $\pi(a \mid s)$ | Land in $s'$   | Reward $r$ | $p(s', r \mid s, a)$ |
| ------ | --------------- | -------------- | ---------- | -------------------- |
| Left   | 0.5             | Safe           | +1         | 1.0 (always)         |
| Right  | 0.5             | Gold           | +2         | 0.8                  |
| Right  | 0.5             | Hole (slipped) | 0          | 0.2                  |

- **Step 1 — Total for each result: $r + \gamma\, v(s')$**
	- Left → Safe: $1 + 0.9 \times 10 = 10$
	- Right → Gold: $2 + 0.9 \times 20 = 20$
	- Right → Hole: $0 + 0.9 \times 0 = 0$
- **Step 2 (Step A) — q for each action: average over its results using $p$**
	- $q(\text{Start}, \text{Left}) = 1.0 \times 10 = \mathbf{10}$
	- $q(\text{Start}, \text{Right}) = 0.8 \times 20 + 0.2 \times 0 = \mathbf{16}$
	- This is exactly the jump example from Part 0: probability × number, then add
- **Step 3 (Step B) — v: average of the q's using $\pi$**
	- $v(\text{Start}) = 0.5 \times 10 + 0.5 \times 16 = 5 + 8 = \mathbf{13}$
	- Meaning: from Start, following this 50/50 policy, on average I collect 13
- **The same calculation written in the exact shape of the equation:**
	- $v(\text{Start}) = \underbrace{0.5}_{\pi(L)} \times \underbrace{\big(1.0 \times [1 + 0.9 \times 10]\big)}_{q(L) = 10} + \underbrace{0.5}_{\pi(R)} \times \underbrace{\big(0.8 \times [2 + 0.9 \times 20] + 0.2 \times [0 + 0.9 \times 0]\big)}_{q(R) = 16} = 13$
- **Optimality version (act perfectly):** $v_*(\text{Start}) = \max(10, 16) = \mathbf{16}$ → best action = **Right**
- **Code (same thing):**
```python
gamma = 0.9
v_next = {'Safe': 10, 'Gold': 20, 'Hole': 0}     # pretend known (normally hidden)
policy = {'Left': 0.5, 'Right': 0.5}             # pi(a|s)
results = {                                      # p(s', r | s, a): list of (prob, s', r)
    'Left':  [(1.0, 'Safe', 1)],
    'Right': [(0.8, 'Gold', 2), (0.2, 'Hole', 0)],
}

v = 0
for a in policy:                                 # Σ over actions
    q = 0
    for prob, s_next, r in results[a]:           # Σ over results of THIS action
        q = q + prob * (r + gamma * v_next[s_next])   # p × (r + γ v(s'))
    print(a, "q =", q)                           # Left 10.0, Right 16.0
    v = v + policy[a] * q                        # π × q
print("v(Start) =", v)                           # 13.0
```

---

## Part 4: The real problem: values are HIDDEN → the DP idea

- In real problems nobody gives us $v(s')$. And $v(A)$ needs $v(B)$, while $v(B)$ needs $v(A)$... chicken and egg 🐔🥚
- **DP trick: guess, then repeat**
	1. Start with a guess: **all values = 0**
	2. For every state, plug the **old guesses** into the right side of the Bellman equation → get **new values**
	3. Repeat step 2 again and again
	4. Stop when the values stop changing (change smaller than a tiny number θ, like 0.0001)
- Why it works: each round, the real rewards ($r$) push information one more step backward. After enough rounds, the guesses become the true hidden values.
- DP needs the model $p(s', r \mid s, a)$ to be **known**. That's why DP works for problems like Gym's FrozenLake (we can read `env.P`).
- **The 3 DP algorithms:**

| Algorithm         | What it does                                                           | Equation used                    |
| ----------------- | ---------------------------------------------------------------------- | -------------------------------- |
| Policy Evaluation | Finds hidden $v_\pi$ for a **given** policy                            | Expectation (average with $\pi$) |
| Policy Iteration  | Evaluate → improve policy → evaluate → ... until policy stops changing | Expectation + "pick best q"      |
| Value Iteration   | Directly finds the best values and best policy                         | Optimality (max)                 |

- **Our tiny world for all of DP: the corridor**
```
  [ A ] — [ B ] — [ G ]
                  goal (episode ends, v(G) = 0 always)
```
- States: A, B, G. Actions: Left, Right. $\gamma = 0.9$
- Moves are not slippery, so every action has **1 result** with $p = 1.0$
- Walking Left from A hits the wall → stay in A
- Reward = **−1 for every step** (so the agent wants to reach G fast)
- Game rules (known):

| State | Action | Land in $s'$ | Reward $r$ | $p$ |
| --- | --- | --- | --- | --- |
| A | Left | A (wall) | −1 | 1.0 |
| A | Right | B | −1 | 1.0 |
| B | Left | A | −1 | 1.0 |
| B | Right | G | −1 | 1.0 |

- **Hidden:** $v(A)$ and $v(B)$. Known: $v(G) = 0$ (game over, no more reward)

---

## Part 5: Policy Evaluation (by hand)

- **Question:** "If I move randomly (50% Left, 50% Right), how good is A and how good is B?"
- **Update rule** = the Bellman expectation equation, with **old** values on the right and **new** values on the left:
	- $V_{new}(s) = \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, V_{old}(s')\big]$
- **Write the equation for each state** (using the game rules table):
	- $V(A) = \underbrace{0.5}_{\pi(L)} \times \underbrace{1.0 \times [-1 + 0.9\,V(A)]}_{q(A,L)} + \underbrace{0.5}_{\pi(R)} \times \underbrace{1.0 \times [-1 + 0.9\,V(B)]}_{q(A,R)}$
	- $V(B) = \underbrace{0.5}_{\pi(L)} \times \underbrace{1.0 \times [-1 + 0.9\,V(A)]}_{q(B,L)} + \underbrace{0.5}_{\pi(R)} \times \underbrace{1.0 \times [-1 + 0.9 \times 0]}_{q(B,R),\ V(G)=0}$
- **Iteration 0 (the guess):** $V(A) = 0,\ V(B) = 0$
- **Iteration 1** (old: A = 0, B = 0)
	- A: $q(A,L) = -1 + 0.9 \times 0 = -1$, $q(A,R) = -1 + 0.9 \times 0 = -1$
		- $V(A) = 0.5 \times (-1) + 0.5 \times (-1) = -1$
	- B: $q(B,L) = -1 + 0.9 \times 0 = -1$, $q(B,R) = -1 + 0 = -1$
		- $V(B) = 0.5 \times (-1) + 0.5 \times (-1) = -1$
- **Iteration 2** (old: A = −1, B = −1)
	- A: $q(A,L) = -1 + 0.9 \times (-1) = -1.9$, $q(A,R) = -1 + 0.9 \times (-1) = -1.9$
		- $V(A) = 0.5 \times (-1.9) + 0.5 \times (-1.9) = -1.9$
	- B: $q(B,L) = -1 + 0.9 \times (-1) = -1.9$, $q(B,R) = -1 + 0 = -1$
		- $V(B) = 0.5 \times (-1.9) + 0.5 \times (-1) = -0.95 - 0.5 = -1.45$
- **Iteration 3** (old: A = −1.9, B = −1.45)
	- A: $q(A,L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(A,R) = -1 + 0.9 \times (-1.45) = -2.305$
		- $V(A) = 0.5 \times (-2.71) + 0.5 \times (-2.305) = -1.355 - 1.1525 = -2.5075$
	- B: $q(B,L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(B,R) = -1$
		- $V(B) = 0.5 \times (-2.71) + 0.5 \times (-1) = -1.355 - 0.5 = -1.855$
- **Keep repeating the same thing** (code does it for us):

| Iteration | V(A) | V(B) | Biggest change |
| --- | --- | --- | --- |
| 0 | 0 | 0 | - |
| 1 | −1.000 | −1.000 | 1.000 |
| 2 | −1.900 | −1.450 | 0.900 |
| 3 | −2.508 | −1.855 | 0.608 |
| 4 | −2.963 | −2.128 | 0.456 |
| 5 | −3.291 | −2.333 | 0.328 |
| 10 | −3.992 | −2.766 | 0.067 |
| 20 | −4.165 | −2.873 | 0.003 |
| 31 | −4.172 | −2.878 | < 0.0001 ✅ stopped changing |

- **Result:** $v_\pi(A) \approx -4.17$, $v_\pi(B) \approx -2.88$ → the **hidden values are now found** ✅
- **Meaning:** moving randomly from A wastes many steps (bumping the wall, going back), so it's very negative. B is closer to G, so less negative.
- **Check:** plug the final values into the equation for B: $0.5 \times [-1 + 0.9 \times (-4.17)] + 0.5 \times [-1] = -2.38 - 0.5 = -2.88$ ✅ left side = right side. That's what "solved the Bellman equation" means.

---

## Part 6: Policy Improvement + Policy Iteration (by hand)

- **Policy Improvement:** "Now that I know the values, can I pick better actions?"
	- Use Step A of the equation to get **q for every action**, using the values from Part 5:
		- $q(A, L) = -1 + 0.9 \times v(A) = -1 + 0.9 \times (-4.17) = -4.75$
		- $q(A, R) = -1 + 0.9 \times v(B) = -1 + 0.9 \times (-2.88) = -3.59$ ← bigger
		- $q(B, L) = -1 + 0.9 \times v(A) = -1 + 0.9 \times (-4.17) = -4.75$
		- $q(B, R) = -1 + 0.9 \times v(G) = -1 + 0 = -1$ ← bigger
	- New policy = in each state, **pick the action with the biggest q** (greedy):
		- A → Right (100%), B → Right (100%)
- **Policy Iteration = repeat (Evaluate → Improve) until the policy stops changing**
	- **Round 1**
		- Evaluate random policy → $v(A) = -4.17$, $v(B) = -2.88$ (Part 5)
		- Improve → new policy: A → Right, B → Right
	- **Round 2**
		- Evaluate new policy. Now $\pi(\text{Right}) = 1$ and $\pi(\text{Left}) = 0$, so the equation becomes:
			- $V(A) = 1 \times [-1 + 0.9\,V(B)] + 0 \times [\dots] = -1 + 0.9\,V(B)$
			- $V(B) = 1 \times [-1 + 0.9 \times 0] = -1$
		- Iterations (start from 0, 0):
			- Iteration 1: $V(A) = -1 + 0.9 \times 0 = -1$, $V(B) = -1$
			- Iteration 2: $V(A) = -1 + 0.9 \times (-1) = -1.9$, $V(B) = -1$
			- Iteration 3: $V(A) = -1.9$, $V(B) = -1$ → no change, stop ✅
		- Improve with $v(A) = -1.9$, $v(B) = -1$:
			- $q(A, L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(A, R) = -1 + 0.9 \times (-1) = -1.9$ ← Right still best
			- $q(B, L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(B, R) = -1$ ← Right still best
		- Policy didn't change → **stop. This is the best policy** ✅
- **Final:** A → Right, B → Right, with $v(A) = -1.9$, $v(B) = -1$
- **Sanity check:** from A the best path is A → B → G, 2 steps: $-1 + 0.9 \times (-1) = -1.9$ ✅

---

## Part 7: Value Iteration (by hand)

- **Idea:** skip the separate "evaluate fully, then improve". Use the **optimality equation** (max) directly as the update:
	- $V_{new}(s) = \max_a \sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, V_{old}(s')\big] = \max_a q(s,a)$
	- In 2 steps: compute $q$ for every action (Step A), then take the **biggest** one
- **Iteration 0:** $V(A) = 0,\ V(B) = 0$
- **Iteration 1** (old: A = 0, B = 0)
	- A: $q(A,L) = -1 + 0.9 \times 0 = -1$, $q(A,R) = -1 + 0.9 \times 0 = -1$ → $V(A) = \max(-1, -1) = -1$
	- B: $q(B,L) = -1 + 0.9 \times 0 = -1$, $q(B,R) = -1 + 0 = -1$ → $V(B) = \max(-1, -1) = -1$
- **Iteration 2** (old: A = −1, B = −1)
	- A: $q(A,L) = -1 + 0.9 \times (-1) = -1.9$, $q(A,R) = -1 + 0.9 \times (-1) = -1.9$ → $V(A) = -1.9$
	- B: $q(B,L) = -1 + 0.9 \times (-1) = -1.9$, $q(B,R) = -1 + 0 = -1$ → $V(B) = \max(-1.9, -1) = -1$
- **Iteration 3** (old: A = −1.9, B = −1)
	- A: $q(A,L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(A,R) = -1 + 0.9 \times (-1) = -1.9$ → $V(A) = -1.9$
	- B: $q(B,L) = -1 + 0.9 \times (-1.9) = -2.71$, $q(B,R) = -1$ → $V(B) = -1$
	- Nothing changed → **stop** ✅
- **Note**: **The rule:** the value of an end (terminal) state is **always 0, by definition.** Once the game is over, no more rewards can come, so the “future total” from there is 0. That’s true in every environment.

| Iteration | V(A) | V(B) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | −1 | −1 |
| 2 | −1.9 | −1 |
| 3 | −1.9 | −1 ✅ |

- **Read the best policy:** in the last iteration, which action gave the max?
	- A: Right (−1.9 > −2.71). B: Right (−1 > −2.71)
- Same answer as Policy Iteration, but faster here (3 small rounds)

- **Policy Evaluation vs Value Iteration**

| | Policy Evaluation | Value Iteration |
| --- | --- | --- |
| Question | How good is **this** policy? | What is the **best** value and policy? |
| Step B uses | Average with $\pi$ | Max |
| Gives a policy? | No (need the improvement step) | Yes (the action that gave the max) |
| Corridor result | A = −4.17, B = −2.88 | A = −1.9, B = −1 |

- Note: some code updates V **in place** (uses the new V(A) right away when computing V(B)). It finishes a bit faster and gives the same final answer. I used old values here because it's easier by hand.

---

## Part 8: Code - Practices

### Game 1 FrozenLake-v1

- my understanding of env

    - Action: 0=left, 1=down, 2=Right, 3=UP
    - observation: 4X4 = 16 spaces to be in where each represented by 0-15 numbers
    - Reward schedule: reach goal is 1, hole and frozen is 0
    - when the agent goes into teh hole the episode ends.
    - is_slippery=True, you never slip in opposite direction
        - Left → left, down, up (probability is fixed with value 1/3 always)
        - Down → down, left, right
        - Right → right, down, up
        - Up → up, left, right
- let build on this then you tell me ok
```python
# training
import gym
env= gym.make('FrozenLake-v1',map_name="8x8", is_slippery=True)

discount_factor=0.9
state_value = []
policy = []

for i in range(env.observation_space.n):
    state_value.append(0)
    policy.append(0)  # all take left as action

state= env.reset()
iteration =0
while(True):
    iteration+=1
    change=0
    # Looping for all the state
    for i in range(env.observation_space.n):
        # Looping for all the action in each state
        max_q_value=0
        for j in range(env.action_space.n):
            q_value=0
            all_possible_poition=env.unwrapped.P[i][j]
            for prob, next_state, rewad, is_complete in all_possible_poition:
                q_value+=(rewad+discount_factor*state_value[next_state])*prob
            if q_value > max_q_value:
                max_q_value = q_value
                policy[i]=j
        change = max(change, abs(max_q_value-state_value[i]))  
        state_value[i]=max_q_value
    if change ==0:
        break

print(change, iteration)
print(state_value)
print(policy)

# inference 
# for display video part
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
def display_video(frames, fps=2):
    fig = plt.figure()
    plt.axis("off")
    img = plt.imshow(frames[0])
    def update(i):
        img.set_data(frames[i])
        return [img]
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())


state=env.reset()
done=False
frames = [env.render(mode="rgb_array")]
while not done:
    action=policy[state]
    next_state, reward, done, info = env.step(action)
    frames.append(env.render(mode="rgb_array"))
    state = next_state
env.close()
display_video(frames)
```

- observation after training and inference
	- **Starting values (0, −1, −2):** doesn’t matter. Value iteration reaches the same final values from any start. That’s why −1 changed nothing
	- **Slippery off → straight to the goal:** correct ✅. Without randomness the best path is the direct one.
	- **Slippery on → it wanders:** correct ✅. The policy says “go toward the goal while avoiding holes”, but the ice keeps pushing you elsewhere. So it keeps correcting until it gets there. You can’t control that randomness. Sometimes it slips into a hole even with the best policy, and that’s normal (about 75% wins at best).
	- **Making every step −1:** good instinct, and you saw the problem. If every step costs −1, falling into a hole _ends_ the cost. So the agent learns “jump into the hole quickly” 😄. That’s a classic reward design trap. Here γ = 0.9 already makes it prefer shorter paths, since a reward that comes later is worth less.
---
### Game 2 CliffWalking

- my understanding of env
    - Action: 0,1,2,3
    - observation: 0....47
    - Reward range: -inf to inf
    - when the agent goes into the cliff -100 reward.
- let build on this then you tell me ok
```python 
import gym
env= gym.make('CliffWalking-v0')
state= env.reset()

discount_factor=0.9
state_value = []
policy = []
for i in range(env.observation_space.n):
    state_value.append(0)
    policy.append(0)  # all take left as action, random action ok
    
iteration =0
while(True):
    iteration+=1
    change=0
    # Looping for all the state
    for i in range(env.observation_space.n):
        # Looping for all the action in each state
        max_q_value=-float('inf')
        for j in range(env.action_space.n):
            q_value=0
            all_possible_poition=env.unwrapped.P[i][j]
            for prob, next_state, rewad, is_complete in all_possible_poition:
                if is_complete:
                    q_value+=rewad*prob # becuase we assuem that the state_value for last state is going to be 0 always
                else:
                    q_value+=(rewad+discount_factor*state_value[next_state.item()])*prob
            if q_value >= max_q_value:
                max_q_value = q_value
                policy[i]=j
        change = max(change, abs(max_q_value-state_value[i]))  
        state_value[i]=max_q_value
    if change < 0.0001:
        break
# Display part: CliffWalking (old gym) can only draw TEXT, not images
# so we use mode="ansi", which gives the grid back as a string
# x = agent, C = cliff, T = goal (target), o = normal cell
state = env.reset()
done = False
frames = [env.render(mode="ansi")]       # save the starting grid as text
steps = 0
while not done and steps < 100:          # safety stop: never loop forever
    action = policy[state]
    next_state, reward, done, info = env.step(action)
    frames.append(env.render(mode="ansi"))
    state = next_state
    steps += 1
print("steps taken:", steps)
```

### Game 3 Taxi

- my understanding of env
    - The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.
    - There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue). The taxi starts off at a random square and the passenger at one of the designated locations.
    - Action
        - 0: Move south (down)
        - 1: Move north (up)
        - 2: Move east (right)
        - 3: Move west (left)
        - 4: Pickup passenger
        - 5: Drop off passenger
    - observation: 500 discrete states
    - Passenger locations:
        - 0: Red
        - 1: Green
        - 2: Yellow
        - 3: Blue
        - 4: In taxi
    - Destinations:
        - 0: Red
        - 1: Green
        - 2: Yellow
        - 3: Blue
    - Reward:
        - -1 per step unless another reward is triggered.
        - +20 delivering passenger.
        - -10  executing "pickup" and "drop-off" actions illegally.
- let build on this then you tell me ok

```python
# NEW library: gymnasium (the maintained version of gym)
# install once in your venv:  pip install "gymnasium[toy-text]"
import gymnasium as gym

env = gym.make('Taxi-v4', render_mode="rgb_array")
state, info = env.reset()      # NEW: reset() returns (state, info)


discount_factor=0.9
state_value = []
policy = []
for i in range(env.observation_space.n):
    state_value.append(0)
    policy.append(0)  # all take 0 as action first


iteration =0
while(True):
    iteration+=1
    change=0
    # Looping for all the state
    for i in range(env.observation_space.n):
        # Looping for all the action in each state
        max_q_value=-float('inf')
        for j in range(env.action_space.n):
            q_value=0
            all_possible_poition=env.unwrapped.P[i][j]
            for prob, next_state, rewad, is_complete in all_possible_poition:
                if is_complete:
                    q_value+=rewad*prob # becuase we assuem that the state_value for last state is going to be 0 always
                else:
                    q_value+=(rewad+discount_factor*state_value[next_state])*prob
            if q_value >= max_q_value:
                max_q_value = q_value
                policy[i]=j
        change = max(change, abs(max_q_value-state_value[i]))  
        state_value[i]=max_q_value
    if change < 0.0001:
        break

  

# for display video part
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
def display_video(frames, fps=2):
    fig = plt.figure()
    plt.axis("off")
    img = plt.imshow(frames[0])
    def update(i):
        img.set_data(frames[i])
        return [img]
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())
  
state, info = env.reset()                 # NEW: returns (state, info)
done = False
frames = [env.render()]                   
steps = 0
while not done and steps < 100:           # safety stop: never loop forever
    action = policy[state]
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated        # same meaning as old "done"
    frames.append(env.render())
    state = next_state
    steps += 1
env.close()
print("steps taken:", steps)
display_video(frames)
```