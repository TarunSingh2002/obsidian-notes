---
tags:
  - python
  - asyncio
  - concurrency
---
## Why Async?
- In sync code, if 100 requests arrive at the same time and each call takes 200 ms, a thread pool of 10 threads handles only 10 requests at once. The other 90 wait in line
- Asynchronous programming lets you interleave many such waiting operations on a single thread (or a few), boosting scalability without spinning up more threads

---

## Types of Operations
- **CPU-bound**
	- Tasks that spend most of their time crunching numbers, manipulating data in memory, or doing heavy computation
	- Key trait: they need raw CPU cycles
- **I/O-bound**
	- Tasks that spend most of their time waiting for external systems: reading/writing files, network requests, database queries, disk access, etc.
	- Key trait: the CPU is idle while the OS or network performs the I/O

### Synchronous (Blocking) Code
- Every call that takes time, whether it's I/O or compute, blocks the current thread until it finishes

### Asynchronous (Non-Blocking) Code
- **CPU-bound task**
	- CPU-bound code always blocks until it finishes
- **I/O-bound task**
	- If written asynchronously → the event loop juggles other work until the I/O operation completes
	- If not written asynchronously → the event loop is stuck until it finishes

---

## Event Loop
- The heart of async: a loop that watches for I/O readiness and timers, and schedules callbacks when operations complete
	- Meaning → instead of blocking on a slow operation or a `sleep()`, your program's "engine" asks, "Hey OS, ping me when X is ready or when 3 seconds have passed," then goes on to do other work. As soon as the OS says "I/O is ready" or the timer expires, the loop puts your waiting code back on the queue and runs it
	- **Watches for I/O readiness**: it asks the operating system, "Let me know when any network socket or file is ready to read/write," instead of blocking
	- **Watches timers**: it also keeps track of "wake-me-up-later" alarms (like `sleep()` or timeouts)
	- **Schedules callbacks**: as soon as an I/O event arrives or a timer goes off, it puts the matching piece of your code back on its to-do list and runs it
- Think of it as a tiny scheduler. It's a loop in your code (usually on the main thread) that:
	- Maintains a queue of "ready" tasks
	- Watches for I/O readiness via OS primitives (e.g. epoll/kqueue)
	- Resumes each `await`-ed coroutine when its awaited operation completes

### Does the event loop "execute" the code?
- No, it **manages** it

### Is the event loop a thread?
- No. The event loop isn't a thread. It's a scheduler that runs in your program's main thread
- Why it's not a thread
	- A thread is a separate "worker" that can run code on its own. The event loop doesn't do that. It's a manager that stays in one thread and switches between tasks quickly
- Thread vs. loop
	- You could run multiple event loops, each in its own OS thread, but by default frameworks use one event loop per thread
	- The loop is code that runs on whatever thread you created it in. It isn't a thread itself

---

## Coroutines
- Special functions defined with `async def`. Calling one returns a **coroutine object**; it isn't executed immediately

## Tasks
- Wrappers around coroutines that schedule them on the event loop
- Created via `asyncio.create_task(coro)`

---

## ThreadPoolExecutor

### What is a thread
- A thread is like a worker that performs tasks inside your program
- Each thread runs independently but shares the same memory with the other threads in the program

### What is a thread pool
- A thread pool is a team of reusable workers (threads)
- Key features
	- Instead of creating a new thread every time a task arrives (slow and inefficient), you:
		- Create a fixed number of threads (e.g. 3 workers)
		- Submit tasks to the pool
		- The pool assigns the tasks to its threads
	- Threads work on different tasks, one task per thread at a time
- Working
	1. **Create a pool**: decide how many threads you want in the pool (e.g. 3 threads)
	2. **Submit tasks**: give the pool tasks to perform
	3. **Assign tasks**: each thread picks a task and works on it until it's done, then takes the next available task

### ThreadPoolExecutor vs ProcessPoolExecutor

| Feature        | ThreadPoolExecutor                                       | ProcessPoolExecutor                     |
| -------------- | -------------------------------------------------------- | --------------------------------------- |
| Workers        | Uses threads                                             | Uses processes                          |
| Memory sharing | Threads share the same memory                            | Processes have separate memory          |
| Best for       | I/O-bound tasks (e.g. file, network)                     | CPU-bound tasks (e.g. calculations)     |
| GIL            | Limited by Python's GIL (1 thread runs Python code at a time) | Not limited by the GIL (true parallelism) |
| Overhead       | Lightweight (threads are cheap)                          | More overhead (processes are heavier)   |

### What is the GIL (Global Interpreter Lock)
- Python's GIL ensures that only one thread executes Python code at a time
- So threads can't run CPU-heavy Python code in parallel
- But threads handle I/O tasks efficiently, because the GIL is released while waiting for I/O

---

## Code

### asyncio.to_thread
- A handy function (Python 3.9+) that runs a synchronous (blocking) function, like `unlink` (which deletes a file), in a separate thread so it doesn't block your async program
- Does it use a thread pool executor?
	- Yes, indirectly. Behind the scenes, `asyncio.to_thread` uses the event loop's default thread pool, which is a `ThreadPoolExecutor`. You don't have to set it up yourself; asyncio does it for you

### When to use ThreadPoolExecutor instead?
- When you need control, like setting how many threads to use or doing something fancy, `ThreadPoolExecutor` gives you that power. For most cases, `asyncio.to_thread` is all you need

---
- See also [[Cheat-Sheet-Python-Code]]
