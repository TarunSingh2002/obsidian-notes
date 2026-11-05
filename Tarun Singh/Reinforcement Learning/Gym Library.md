- **What is Gym?**
	- A Python library that gives us ready-made **environments** (games / tasks) to practice RL on.
	- Every environment talks to the agent in the **same way**, so once you learn the 5 basic methods you can use any environment.
	- Examples of environments it has:
		- Classic control → CartPole, Pendulum, MountainCar
		- Atari games → Space Invaders, Breakout, Pong
		- Robotics → robotic arm tasks
	- Gym was made by OpenAI. It is now maintained under the name **Gymnasium** (`import gymnasium as gym`). The course uses old `gym==0.23.0` (see the "Old vs New" section at the bottom).
- **The Big Picture (Agent ↔ Environment loop)**
	- Agent looks at the **state** → picks an **action** → environment gives back a **reward** + the **next state** → repeat.
	- Gym is the "environment" side of this loop.

---

### **The 5 Methods You Must Know**

| Method                                                                         | What it does                     | What it gives back               |
| ------------------------------------------------------------------------------ | -------------------------------- | -------------------------------- |
| `gym.make("Name")` / `Maze()`                                                  | Creates the environment          | An env object                    |
| `env.reset()`                                                                  | Puts the env back at the start   | The starting state               |
| `env.step(action)`                                                             | Does one action in the env       | `next_state, reward, done, info` |
| `env.render()`                                                                 | Draws the current state          | An image (numpy array)           |
| `env.close()`                                                                  | Shuts the env down, frees memory | Nothing                          |
| env.observation_space<br><br>env.action_space<br><br>env.action_space.sample() |                                  |                                  |

- **1. Making the environment**
	- For built-in envs you pass the name as a string to `gym.make`.
	- In the course the instructor built his own env called `Maze`, so we just create it like a normal Python class.
```python
import gym

env = gym.make("CartPole-v1")   # built-in environment
env = Maze()                    # instructor's custom maze (same idea)
```

- **2. `env.reset()` → start a new episode**
	- Puts the agent at the starting position and returns that state so the agent can "see" where it is.
	- Always call this **before** the first step of every episode.
```python
initial_state = env.reset()
print(initial_state)   # (0, 0) → agent starts at row 0, col 0
```

- **3. `env.render()` → see what is happening**
	- Makes a picture of the current state.
	- `mode='rgb_array'` → gives back the picture as a numpy array, which we can show with matplotlib.
```python
from matplotlib import pyplot as plt

frame = env.render(mode='rgb_array')   # picture as a numpy array
plt.axis('off')
plt.imshow(frame)                      # show the picture
```

- **4. `env.step(action)` → do one move**
	- This is the most important method. You give it an action, the env changes, and it gives back **4 things**:

| Returned value | Meaning | Maze example |
|---|---|---|
| `next_state` | Where the agent is now | `(1, 0)` |
| `reward` | Score for that move | `-1.0` |
| `done` | `True` if the episode is over (goal reached) | `False` |
| `info` | Extra debug info (a dict, usually empty) | `{}` |

```python
action = 2                                   # 2 = move DOWN
next_state, reward, done, info = env.step(action)
# next_state → (1, 0)   moved down one row
# reward     → -1.0     every move costs -1 until the exit
# done       → False    not at the exit yet
```

- **5. `env.close()` → clean up**
	- Closes windows and frees memory. Call it when you are finished.
```python
env.close()
```

---

### **Spaces (action_space & observation_space)**

- **Key Idea:** A "space" just describes **what values are allowed**.
	- `env.action_space` → what actions the agent can take
	- `env.observation_space` → what states the agent can see
- **Common space types**

| Space type              | Meaning                                           | Example                                      |
| ----------------------- | ------------------------------------------------- | -------------------------------------------- |
| `Discrete(n)`           | One whole number from `0` to `n-1`                | `Discrete(4)` → actions 0, 1, 2, 3           |
| `MultiDiscrete([a, b])` | A group of whole numbers, each with its own range | `MultiDiscrete([5, 5])` → (row 0-4, col 0-4) |
| `Box(low, high, shape)` | Real numbers (decimals) in a range                | CartPole's state: 4 decimal numbers          |

- **Action space in the Maze**
	- `Discrete(4)` → 4 possible actions

| Action | Meaning |
|---|---|
| 0 | UP |
| 1 | RIGHT |
| 2 | DOWN |
| 3 | LEFT |

- **Observation (state) space in the Maze**
	- `MultiDiscrete([5 5])` → state is `(row, column)`, each from 0 to 4.
	- So there are 5 × 5 = **25 states** in total.
	- The goal (exit) is at `(4, 4)`, bottom-right corner.
```python
print(env.action_space)        # Discrete(4)
print(env.observation_space)   # MultiDiscrete([5 5])
```

- **`.sample()` → pick a random valid value**
	- Super handy for a random agent: it gives a random action that is always allowed.
```python
action = env.action_space.sample()        # e.g. 3 (LEFT)
state  = env.observation_space.sample()   # e.g. [2 4] (random cell)
```

- **How the instructor defines spaces inside his own env** (good to recognise)
```python
from gym import spaces

class Maze(gym.Env):
	def __init__(self):
		super().__init__()
		self.action_space = spaces.Discrete(n=4)                # 4 actions
		self.observation_space = spaces.MultiDiscrete([5, 5])   # (row, col)
```
- **Intuition:** If you build your own env, you just inherit from `gym.Env` and write `reset`, `step`, `render`, `close` + set the two spaces. That's it, now it works like any Gym env.

---

### **Trajectory vs Episode**

- **Trajectory:** any sequence of moves → state, action, reward, state, action, reward ...
	- Can start and stop anywhere (e.g. just 3 moves).
- **Episode:** a trajectory that goes from the **start** all the way to the **end** (until `done = True`).

- **Trajectory of 3 moves** → use a `for` loop (fixed number of steps)
```python
env = Maze()
state = env.reset()
trajectory = []
for _ in range(3):                                     # only 3 moves
	action = env.action_space.sample()                 # random action
	next_state, reward, done, info = env.step(action)
	trajectory.append([state, action, reward, done, next_state])
	state = next_state                                 # move forward
env.close()
```

- **Full episode** → use a `while not done` loop (run until the goal)
```python
env = Maze()
state = env.reset()
episode = []
frames = [env.render(mode='rgb_array')]
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    # Capture the updated maze
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    episode.append([state, action, reward, done, next_state])
    state = next_state
env.close()
display_video(frames)
```
- **This `while not done` loop is the heart of every RL program.** Later algorithms only change *how the action is chosen*.

---

### **Reward & Return in Code**

- **Reward:** the number you get back from **one** `step()`. In the Maze it is always `-1` until you reach the exit → this pushes the agent to find the exit **fast**.
- **Return (G):** the **total** of all rewards in an episode, where later rewards count a bit less (multiplied by gamma = 0.99 again and again).
```python
env = Maze()
state = env.reset()
done = False
gamma = 0.99      # discount factor: future rewards count slightly less
G_0 = 0           # total return from the start
t = 0             # step counter
while not done:
	action = env.action_space.sample()
	_, reward, done, _ = env.step(action)   # _ = "I don't need this value"
	G_0 += gamma ** t * reward               # add discounted reward
	t += 1
env.close()
```
- **Intuition:** More steps to reach the exit → more `-1`s → a more negative return. A good agent = higher (less negative) return.

---

### **Policy in Code**

- **Policy:** the agent's "brain" → given a state, how likely is each action?
- In code: a function that takes a `state` and returns a list of probabilities (one per action, adds up to 1).
```python
import numpy as np

def random_policy(state):
	return np.array([0.25] * 4)   # [0.25, 0.25, 0.25, 0.25] → every action equally likely

probs = random_policy(state)
action = np.random.choice(range(4), p=probs)   # pick an action using those probabilities
```

---

### **The Full Template (put it all together)**

- Every RL program you write will look like this. Only the `policy` changes as you learn better algorithms.
```python
env = Maze()                     # 1. make env
state = env.reset()              # 2. start episode
done = False
frames = [env.render(mode="rgb_array")]

while not done:
	action_probs = policy(state)                              # 3. ask policy
	action = np.random.choice(range(4), p=action_probs)       #    pick action
	next_state, reward, done, info = env.step(action)         # 4. do the action
	frames.append(env.render(mode="rgb_array"))               # 5. save a picture
	state = next_state                                        # 6. move on

env.close()                      # 7. clean up
```

---

### **Old Gym (course) vs New Gymnasium**

- The course uses `gym==0.23.0`. If you install the latest library you will get **Gymnasium**, which changed a few things:

| Thing | Old `gym` (course) | New `gymnasium` |
|---|---|---|
| Import | `import gym` | `import gymnasium as gym` |
| `reset()` returns | `state` | `state, info` |
| `step()` returns | `state, reward, done, info` (4 values) | `state, reward, terminated, truncated, info` (5 values) |
| Rendering | `env.render(mode='rgb_array')` | `gym.make("CartPole-v1", render_mode="rgb_array")` then `env.render()` |

- **`terminated` vs `truncated`:**
	- `terminated` → episode ended naturally (reached goal / fell over).
	- `truncated` → episode was cut off (time limit hit).
	- Old `done` = `terminated or truncated`.
```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
state, info = env.reset()
done = False
while not done:
	action = env.action_space.sample()
	state, reward, terminated, truncated, info = env.step(action)
	done = terminated or truncated      # same meaning as old "done"
env.close()
```
