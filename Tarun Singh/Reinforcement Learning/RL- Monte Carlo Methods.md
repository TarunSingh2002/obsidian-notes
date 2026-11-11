- This is a family of methods that learn the optimal state-values or q-values based on samples of experience collected by the agent while interacting with the environment.
- **Big difference from DP:** DP needed the rules table $p(s', r \mid s, a)$ (`env.unwrapped.P`). Monte Carlo (MC) does **not**. It just **plays** the game and learns from what happened.
- how
	- 1. Play one **full episode** (start → end) with the current policy $\pi$
	- 2. Write down what happened: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$ (the trajectory)
	- 3. For every state visited, compute the **Return** $G_t$ from that moment until the end
	- 4. Value of a state = **average** of all the returns we have collected from that state
	- 5. Repeat for many episodes → the averages get more and more accurate

---

## **Expected value = Σ (probability × value)? Or = average (1/N × Σ)?**

- **Both are correct.** They are two ways to get the same thing:

| Way                           | Formula                                               | When you can use it                                                                     |
| ----------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Exact** (what we did in DP) | $\mathbb{E}[X] = \sum_x P(x) \times x$                | You **know** the probabilities                                                          |
| **Estimate** (what MC does)   | $\mathbb{E}[X] \approx \frac{1}{N}\sum_{k=1}^{N} x_k$ | You **don't know** the probabilities, but you can try many times and write down results |

- **Dice example**
	- Exact way: $\frac{1}{6}(1+2+3+4+5+6) = 3.5$ (we know every face has chance 1/6)
	- Estimate way: roll it many times and average the results
- So "value of a state = **expected** return" and "value of a state = **average** return" mean the same thing. "Expected" is the true answer, and "average of many tries" is how we estimate it when we don't know the probabilities.

## **Law of large numbers (simple)**

- **One line:** the more times you try, the closer your average gets to the true expected value.
- I rolled a dice on the computer (real numbers):

| Number of rolls | Average of results | True expected value |
| --- | --- | --- |
| 5 | 3.2 | 3.5 |
| 10 | 3.1 | 3.5 |
| 100 | 3.34 | 3.5 |
| 1,000 | 3.435 | 3.5 |
| 100,000 | 3.497 | 3.5 |

- With few rolls the average is off (luck matters). With many rolls the luck cancels out → average ≈ 3.5
- The slide's formula $P\big(\lim_{n \to \infty} \bar{G}_s = v_\pi(s)\big) = 1$ just says: "if you collect infinitely many returns from state $s$, their average **will** become exactly $v_\pi(s)$"
- **In MC:** more episodes → more returns per state → more accurate $V(s)$. That's why MC needs **lots** of episodes.

---

## **The important equations**

- **Return (finite episode, ends at time T)**
	- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$
	- Same as before, but it **stops at the end of the episode** (T = last time step)
	- Easy way to compute it: go **backward** from the end using $$G_t = R_{t+1} + \gamma G_{t+1}$$

- **State value**

|                                                | Meaning                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$  | The **true** value (definition, same as DP)                            |
| $V_\pi(s) = \frac{1}{N}\sum_{k=1}^{N} G_{s_k}$ | Our **estimate**: average of the N returns we collected from state $s$ |

- **Action value (Q)**

| | Meaning |
| --- | --- |
| $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$ | The **true** Q value |
| $Q_\pi(s,a) = \frac{1}{N}\sum_{k=1}^{N} G_{s,a_k}$ | Our **estimate**: average of the returns we collected **after doing action a in state s** |

- Small letters ($v, q$) = true value. Capital letters ($V, Q$) = our estimate from experience.

---

## **Important : Example: MC by hand (maze, −1 per step, γ = 0.9, ε = 0.1)**

- **The maze**
```
  [ A ] — [ B ] — [ C ] — [ Goal ]
    |       |
  [ D ] ————
```
- States: A, B, C, D, Goal. Every move gives **−1**. Reaching Goal ends the episode.
- We **don't** know the rules ($p$). The agent plays, we write down what happened.

### **What we are going to do (same as code)**

- Keep only **2 tables**, both the same size (rows = states, columns = actions):
	- **Q table** = the value of each (state, action)
	- **N table** = how many times each (state, action) was updated (this is the n in 1/n)
- **No V table needed:** to pick an action we only look at Q. (If ever needed: V(s) = max of Q's row)
- **No policy table needed:** policy = "in this state take the max of its Q row" (+ ε)
- Every episode:
	- **Policy evaluation** = play 1 episode (with ε), go backward, compute G, update Q with the 1/n formula
	- **Policy improvement** = in every state, best action = max of its Q row
- Then next episode, again evaluate → improve → ...

### **Rules**

- **Return:** go backward, $G_t = R_{t+1} + \gamma\, G_{t+1}$, the last move's G = last reward
	- G = return of **this one episode** only. Q = average of all the G's = expected return
- **ε-greedy (ε = 0.1):** at **every step of every episode** (episode 1 too) pick random $p$
	- $p < 0.1$ → explore → random action
	- else → exploit → max of the Q row
- **1/n formula (used for EVERY update, also the first one):** $Q_{new} = Q_{old} + \frac{1}{n}(G - Q_{old})$
	- Step 1: N[s][a] = N[s][a] + 1 → this is n
	- Step 2: put $Q_{old}$, n and G in the formula
- **Why the first time also uses the formula:** in code Q and N both start at **0**. First time: n = 1, old = 0 → $0 + \frac{1}{1}(G - 0) = G$. No special case.

> **⭐ IMPORTANT: same (state, action) 2 times in one episode → use only the FIRST one's G**
> - Update that Q **only once** per episode, with the G from its **first** time (smallest t)
> - So N goes up by **1 only**, not 2
> - Easy way: go backward from the end, every time you see the same (state, action) again just **overwrite** its G. When you reach t = 0, the G left is the first-time G. Then use the formula once.
> - This is called **first-visit MC**

### **Start**

| Q table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 0 |

| N table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 0 |
| B | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 0 |

- **Start policy (just a guess):** A: Down, B: Right, C: Right, D: Right

---

### **Episode 1**

#### **Evaluation:** every p ≥ 0.1 → exploit every step. Path A → D → B → C → Goal

| t | State | p | Explore / exploit | Action | Reward | G (backward) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | 0.62 | exploit | Down | −1 | $-1 + 0.9 \times (-2.71) = -3.439$ |
| 1 | D | 0.47 | exploit | Right | −1 | $-1 + 0.9 \times (-1.9) = -2.71$ |
| 2 | B | 0.85 | exploit | Right | −1 | $-1 + 0.9 \times (-1) = -1.9$ |
| 3 | C | 0.33 | exploit | Right | −1 | $-1$ |

- **Q update**
	- Q(A,Down): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-3.439 - 0) = -3.439$
	- Q(D,Right): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-2.71 - 0) = -2.71$
	- Q(B,Right): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-1.9 - 0) = -1.9$
	- Q(C,Right): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-1 - 0) = -1$

#### **Improvement:** max of each Q row (only tried actions)

- A: only Down tried → **Down** (no change)
- B: Right, C: Right, D: Right (no change)

#### **Tables after Episode 1**

| Q table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | 0 | −3.439 | 0 | 0 |
| B | −1.9 | 0 | 0 | 0 |
| C | −1 | 0 | 0 | 0 |
| D | −2.71 | 0 | 0 | 0 |

| N table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | 0 | 1 | 0 | 0 |
| B | 1 | 0 | 0 | 0 |
| C | 1 | 0 | 0 | 0 |
| D | 1 | 0 | 0 | 0 |

---

### **Episode 2**

#### **Evaluation:** at A, p = 0.05 < 0.1 → **explore** → Right. Path A → B → C → Goal

| t | State | p | Explore / exploit | Action | Reward | G (backward) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | **0.05** | **explore** | Right (random) | −1 | $-1 + 0.9 \times (-1.9) = -2.71$ |
| 1 | B | 0.71 | exploit | Right | −1 | $-1 + 0.9 \times (-1) = -1.9$ |
| 2 | C | 0.40 | exploit | Right | −1 | $-1$ |

- **Q update**
	- Q(A,Right): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-2.71 - 0) = -2.71$
	- Q(B,Right): n = 1 + 1 = 2 → $-1.9 + \frac{1}{2}(-1.9 - (-1.9)) = -1.9 + \frac{1}{2}(0) = -1.9$
	- Q(C,Right): n = 1 + 1 = 2 → $-1 + \frac{1}{2}(-1 - (-1)) = -1 + \frac{1}{2}(0) = -1$

#### **Improvement:** max of each Q row (only tried actions)

- A: Right (−2.71) vs Down (−3.439) → **Right** (changed! was Down)
- B: Right, C: Right, D: Right (no change)
- Without the random move, A → Right would never be tried and the policy would be stuck on Down. **That's why ε is needed.**

#### **Tables after Episode 2**

| Q table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | −2.71 | −3.439 | 0 | 0 |
| B | −1.9 | 0 | 0 | 0 |
| C | −1 | 0 | 0 | 0 |
| D | −2.71 | 0 | 0 | 0 |

| N table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | 1 | 1 | 0 | 0 |
| B | 2 | 0 | 0 | 0 |
| C | 2 | 0 | 0 | 0 |
| D | 1 | 0 | 0 | 0 |

---

### **Episode 3 (same state + same action 2 times)**

#### **Evaluation:** at B, p = 0.06 < 0.1 → **explore** → Left (goes back to A). Path A → B → A → B → C → Goal

| t | State | p | Explore / exploit | Action | Reward | G (backward) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | 0.66 | exploit | Right | −1 | $-1 + 0.9 \times (-3.439) = -4.0951$ |
| 1 | B | **0.06** | **explore** | Left (random) | −1 | $-1 + 0.9 \times (-2.71) = -3.439$ |
| 2 | A | 0.52 | exploit | Right | −1 | $-1 + 0.9 \times (-1.9) = -2.71$ |
| 3 | B | 0.91 | exploit | Right | −1 | $-1 + 0.9 \times (-1) = -1.9$ |
| 4 | C | 0.28 | exploit | Right | −1 | $-1$ |

- **⭐ (A, Right) shows up 2 times: t = 0 and t = 2**
	- Going backward: at t = 2 its G = −2.71, then at t = 0 overwrite → G = **−4.0951**
	- Use only the **first** one (t = 0) → G = −4.0951. Ignore t = 2.
	- Update Q(A,Right) **once**, N(A,Right) goes up by **1** (1 → 2, not 3)
- (B, Left) at t = 1 and (B, Right) at t = 3 are **different actions**, so they are different pairs. Each is updated normally.
- **Q update**
	- Q(A,Right): n = 1 + 1 = 2 → $-2.71 + \frac{1}{2}(-4.0951 - (-2.71)) = -2.71 + \frac{1}{2}(-1.3851) = -2.71 - 0.6926 = -3.4026$
	- Q(B,Left): n = 0 + 1 = 1 → $0 + \frac{1}{1}(-3.439 - 0) = -3.439$
	- Q(B,Right): n = 2 + 1 = 3 → $-1.9 + \frac{1}{3}(-1.9 - (-1.9)) = -1.9 + 0 = -1.9$
	- Q(C,Right): n = 2 + 1 = 3 → $-1 + \frac{1}{3}(-1 - (-1)) = -1 + 0 = -1$

#### **Improvement:** max of each Q row (only tried actions)

- A: Right (−3.4026) vs Down (−3.439) → **Right** (no change, but very close now)
- B: Right (−1.9) vs Left (−3.439) → **Right**. Exploration tried Left and found it is worse ✅
- C: Right, D: Right

#### **Tables after Episode 3**

| Q table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | −3.4026 | −3.439 | 0 | 0 |
| B | −1.9 | 0 | −3.439 | 0 |
| C | −1 | 0 | 0 | 0 |
| D | −2.71 | 0 | 0 | 0 |

| N table | Right | Down | Left | Up |
| --- | --- | --- | --- | --- |
| A | **2** | 1 | 0 | 0 |
| B | 3 | 0 | 1 | 0 |
| C | 3 | 0 | 0 | 0 |
| D | 1 | 0 | 0 | 0 |

- **Final policy (max of each row):** A: Right, B: Right, C: Right, D: Right

### **Notes**

- **"Only tried actions":** in code untried actions sit at 0. Here every real return is negative, so 0 would look "best" and argmax would pick an untried action. In code that just means extra exploration. By hand we only compare tried actions to keep it simple.
- **Why Q(A, Right) went down:** the random Left at B made Episode 3 long. Random moves also get mixed into the values. Many more episodes fix this (law of large numbers).
- No $p(s', r \mid s, a)$ anywhere. We only used rewards we actually got.
- Fixed maze → $\frac{1}{n}$. Changing maze → constant α like 0.1 (see Doubt 1).

---

## **MC vs DP**

| | Dynamic Programming | Monte Carlo |
| --- | --- | --- |
| Needs the model $p$ (`env.P`)? | Yes | **No**, learns from playing |
| How it gets values | Uses other states' values (bootstrapping) | Uses real returns from full episodes |
| Which states it updates | **All** states, every sweep | Only states the agent **visits** (focuses on useful ones) |
| One state's value depends on others? | Yes | No, each state is estimated on its own |
| Weak point | Needs the rules | Must wait until the episode **ends**, needs **many** episodes |

---

## **Control: finding the best policy with MC**

- **Same loop as Policy Iteration (called Generalized Policy Iteration):**
	- $\pi_0 \to Q_{\pi_0} \to \pi_1 \to Q_{\pi_1} \to \cdots \to Q_{\pi_*} \to \pi_*$
	- **Policy evaluation:** play episodes and update the Q table with average returns
	- **Policy improvement:** in each state, pick the action with the highest Q
	- Worked by hand in "Policy evaluation and policy improvement in this example" (in the example section above)

- **Why Q and not V?** (very important)
	- In DP we improved the policy with: $\pi'(s) = \arg\max_a \sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, v_\pi(s')\big]$
		- **What this scary equation simply means:** the part $\sum_{s',r} p(s', r \mid s, a)\big[r + \gamma\, v_\pi(s')\big]$ is just **q(s, a)** (Step A from the DP note), computed using $p$. So the whole thing is: "for each action, compute its q using $p$, then pick the action with the biggest q"
		- In other words: $\pi'(s) = \arg\max_a q_\pi(s,a)$, where DP got q from $p$
	- That needs $p$, which MC doesn't have ❌
	- **MC's trick:** get q **without** $p$, using the "estimate" way of expected value: play, collect returns, average them → $Q(s,a) = \frac{1}{N}\sum G$. Then pick the biggest, same as DP
	- With Q we can skip it: $\pi'(s) = \arg\max_a q_\pi(s,a)$ ✅ → just look in the table, pick the biggest
	- Q already "contains" the environment's randomness, because it's the average of real returns after taking that action
	- So in MC we keep a **table of Q(s, a)** instead of a table of V(s)

- **The exploration problem**
	- At the start, Q estimates are bad (few episodes)
	- If the best action got a low estimate by bad luck, a greedy policy will **never** pick it again → its estimate never gets fixed
	- Fix: make sure **every action gets tried sometimes**
		- **Option 1: Exploring starts:** every episode starts from a random state with a random first action. Not realistic in most real tasks.
		- **Option 2: Stochastic policy:** every action always has probability > 0 (like ε-greedy from the bandit note). More realistic and easier ✅
	- Two ways to use a stochastic policy:
		- **On-policy:** the same policy explores and gets improved
		- **Off-policy:** one policy explores, a different policy gets improved

---

## **Cheat sheet**

- MC = play full episodes → compute returns → average them
- Expected value can be computed exactly ($\sum P \times x$) or estimated ($\frac{1}{N}\sum x$). MC estimates.
- Law of large numbers: more episodes → average → true value
- No model needed, so we learn **Q(s,a)** and improve with $\arg\max_a Q(s,a)$
- Must keep exploring (exploring starts or a stochastic policy like ε-greedy)

---

## **Important section: Doubt section**

### **Doubt 1: Does ε-greedy work when the bandit is fixed AND when it keeps changing?**
- First ε-greedy Formula:
```python  

# Pseudo code
eps=0.05 # 5% epsilon value
p=random() # 0 to 1
if p< eps:
	Go with random action [mean explore]
else:
	Go with current-best action [mean exploit]

# Only for the action you just played, do this (not every action)
# new_mean = old_mean + (1/n) * (new_reward - old_mean)
self.mean = self.mean + (1.0 / self.n) * (x - self.mean)
# Case 2 (changing): use a constant alpha instead of 1/n
# self.mean = self.mean + 0.1 * (x - self.mean)

```
- **The two cases:**
	- **Case 1: fixed (stationary) bandit**
		- 3 slot machines with win chances **10%, 20%, 50%**
		- These chances **never change**. Machine 3 is always the best.
	- **Case 2: changing (non-stationary) bandit**
		- Same 3 machines, but every 30 minutes their win chances get shuffled
		- First 30 min: 10%, 20%, **50%** → machine 3 is best
		- Next 30 min: **50%**, 10%, 20% → now machine 1 is best
		- Next 30 min: something else again...
- **Short answer:** ε-greedy is fine in both cases. But in Case 2 it only works if you **also** update the estimate (the agent never stops learning while it plays, so its estimates keep adjusting by themselves; no need to restart training every 30 min) with a constant $\alpha$ like (0.1) instead of $\frac{1}{N}$. Because 1/N shrink over the period of time which mean new observation has less contribution which we don't want.
- **Remember:** Case 1 → ε-greedy + $\frac{1}{N}$ (or $\alpha$). Case 2 → ε-greedy + constant $\alpha$.
### **Doubt 2: If the environment keeps changing, does RL stop working?**

- **My setup (what I did in FrozenLake DP):**
	- Train on one map (4x4 or 8x8) → get a policy → use that policy on the **same** map
	- Holes and goal are in the **same places** during training and during prediction ✅ → the policy works
- **Short answer:** I'm right. With DP and Monte Carlo as I've used them so far, **if the map changes, the old policy is useless and I must train again for the new map.** A policy only works on the map it was trained on.

- **Why the policy breaks (the real reason):**
	- In FrozenLake the state is **only my position** (a number 0–15). The hole positions are **not** in the state.
	- So the policy is just a list: "in square 5, go Left". It doesn't know **why**. It only memorised what worked on that one map.
	- On a new map, square 5 may now be next to a hole, but the policy can't see that. It still says "go Left" → falls in.
	- **Rule:** the agent can only handle changes it can **see in its state**.

- **3 situations:**

| Situation | Example | What to do |
| --- | --- | --- |
| **A. Map never changes** | Same FrozenLake map always | Train once, use forever ✅ (what I did) |
| **B. Map changes sometimes** | Holes move once a day | Train again for each new map ❌ old policy is useless. (MC can keep learning on the new map with a constant $\alpha$, but it still needs many new episodes before the policy is good again) |
| **C. Map changes every episode** | Holes spawn in new places every game | Training per map won't work, there's no time. Need a different idea 👇 |

- **How to solve situation C (truly changing map):**
	- **Key idea:** put the map **inside the state**, so the agent can **see** where the holes are.
		- Old state: "I'm at square 5"
		- New state: "I'm at square 5 **and** the holes are at 3, 7, 11"
	- Now the agent can learn a **general rule** like "don't step next to a hole", instead of memorising one map. This is called **generalisation**.
	- **Option 1: DP per map (only if I know the rules):** at the start of each episode, read the new map, build its $p$, run value iteration (under a second for 4x4), then play. This is **planning**: it works, but it isn't "learning" anything that carries over to the next map.
	- **Option 2: Q table with the map in the state:** works in theory, but the table explodes. A 4x4 map where each square can be a hole or not = $2^{16} = 65{,}536$ maps × 16 positions ≈ **1 million states** × 4 actions. Too big to visit every state with MC ❌
	- **Option 3: Deep RL (the real solution):** a **neural network** takes the whole map (like an image) as input and outputs the Q value of each action. It learns patterns ("hole next to me → don't go there") that work on maps it has **never seen**. This is how agents for games with random levels are trained. It's later in the course (Deep Q-Learning, PyTorch).

- **Remember:**
	- DP and MC (with a table) learn **one fixed world**. New world → train again.
	- To handle a world that changes every time, the agent must **see** the change (put it in the state), and usually needs a **neural network** to generalise.

### **Doubt 3: Is the Q table rows = states, columns = actions? Can it get too big?**

- **Short answer:** yes to both.
- **Shape:** one row per state, one column per action

| State | Action 0 | Action 1 | Action 2 | Action 3 | Best action (argmax) |
| --- | --- | --- | --- | --- | --- |
| 0 | −3.2 | **−2.1** | −4.0 | −3.5 | 1 |
| 1 | **−1.0** | −2.5 | −1.9 | −3.0 | 0 |
| ... | ... | ... | ... | ... | ... |

- **Picking the action:** in the row of the current state, find the column with the **biggest** number. That column index is the action. That's what $\arg\max_a Q(s,a)$ means.
- **Size:**
	- Number of actions is usually small (2 to 20 columns)
	- Number of states can be huge:

| Game | States | Q table size | OK? |
| --- | --- | --- | --- |
| FrozenLake 4x4 | 16 | 16 × 4 = 64 | ✅ |
| Taxi | 500 | 500 × 6 = 3,000 | ✅ |
| 1,000 states, 10 actions | 1,000 | 10,000 | ✅ still fine |
| CartPole | infinite (decimal numbers) | impossible | ❌ |
| Atari games | way too many to ever store or visit | impossible | ❌ |

- **Fix for huge games (later in the course):** replace the table with a **neural network**. Input = the state, output = one Q value per action. Same idea (pick the biggest Q), just no table. This is **Deep Q-Learning**, and it's where PyTorch starts.

---

##  ** Code Example **

### On-policy MC - FrozenLake

- **My understanding of env**
	- Action: 0=left, 1=down, 2=Right, 3=UP
	- observation: 4*4 = 16 spaces to be in where each represented by 0-15 numbers
	- Reward schedule: reach goal is 1, hole and frozen is 0
	- when the agent goes into the hole the episode ends.
	- is_slippery=True, you never slip in opposite direction
		- Left → left, down, up [probability is fixed with value 1/3 always]
		- Down → down, left, right
		- Right → right, down, up
		- Up → up, left, right
- **Type:** on-policy first-visit MC control (the same ε-greedy policy plays AND its G's update Q). Testing with pure greedy at the end does **not** make it off-policy.
- **Result:** Success rate **73.30%** on slippery 4x4 (best possible is ~73%, so this is the optimal policy)



```python
import gymnasium as gym
import random
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="rgb_array")

discount_factor = 0.9
epsilon = 1.0
n_table = []
q_table = []
for state_i in range(env.observation_space.n):
    q_row, n_row = [], []
    for action_j in range(env.action_space.n):
        q_row.append(None) # for q value table
        n_row.append(0) # for n table
    n_table.append(n_row)
    q_table.append(q_row)

def greedy_action(q_row):
    max_q = float('-inf')
    num_actions = len(q_row)
    best_action = 0
    for a in range(num_actions):
        if q_row[a] is not None and max_q < q_row[a]:
            max_q = q_row[a]
            best_action = a
    if max_q == float('-inf'):
        return random.randint(0, num_actions-1) # this will only happen for the first time taht complete row is None
    return best_action

episode_num = 0

while(episode_num < 1000000):
    episode_num += 1
    if epsilon > 0.05:
        epsilon -= 5e-7
    # starting a episode
    episode = [] # each time epiode reste to []
    # run until the episode ends
    state, info = env.reset()
    while(True):
        episode.append(state)
        p_random = random.random()
        if p_random > epsilon:
            action = greedy_action(q_table[state])
            # go with exploite ok
        else:
            action = random.randint(0, env.action_space.n-1)
            # go with explore action

        episode.append(action)
        # now we will take the action
        state, reward, terminated, truncated, info = env.step(action)

        episode.append(reward)

        if terminated or truncated:
            break

    # now the epiode is over we will calucate the G for each sate ok
    # epiose will be like s0-a0-r1-s1-a1-r2

    episode_len = len(episode)
    first_visit_G = {}
    idx = episode_len-1
    G_next = 0
    while(idx >= 0):
        state_action = ( episode[idx-2], episode[idx-1] )
        reward = episode[idx]
        idx -= 3
        G = reward + discount_factor*G_next
        first_visit_G[state_action] = G
        G_next = G

    # now updating the n matrics and q value matrics
    for key, G_value in first_visit_G.items():
        state_now , action_now = key
        n_table[state_now][action_now] += 1
        q_old = q_table[state_now][action_now]
        if q_old == None:
            q_old = 0
        q_new = q_old + (G_value-q_old)/n_table[state_now][action_now]
        q_table[state_now][action_now] = q_new

    # no need to update policy, as we have no policy amtrics
    # the function greedy_action will action need to taken if we want max q_value

# now generate policy
policy = []
for state_i in range(env.observation_space.n):
    policy.append(greedy_action(q_table[state_i]))
# policy = [0, 3, 0, 3, 0, 1, 0, 2, 3, 1, 0, 0, 0, 2, 1, 1]
# (states 5, 7, 11, 12 are holes and 15 is goal, their action does not matter)

success_count = 0
episodes = 1000
for _ in range(episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated or truncated:
            done = True
            if reward == 1:
                success_count += 1
success_rate = success_count / episodes
print(f"Success rate: {success_rate:.2%}")   # Success rate: 73.30%
env.close()
```