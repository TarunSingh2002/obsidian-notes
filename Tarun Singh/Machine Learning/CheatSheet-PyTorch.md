---
tags:
  - deepLearning
  - pytorch
  - cheatSheet
---
## Tensors

- Types of tensor based on dimensions

| Name | Dim | Example |
| ---- | --- | ------- |
| scalar | 0 D | `torch.tensor(4.0)` |
| vector | 1 D | `torch.tensor([1,2,3])` |
| matrix | 2 D | `torch.tensor([[1,2,3],[4,5,6]])` |
| 3 D tensor | 3 D | image (H, W, channels) |
| ... n D tensor | n D | batch of images (N, C, H, W) |

### Creating a tensor
```python
import torch

torch.tensor([[1,2,3],[4,5,6]])          # from a python list
torch.tensor([1,2,3], dtype=torch.float64)   # with a specific datatype
torch.tensor([1,2,3], dtype=torch.int32)

torch.zeros(2,3)                # all 0
torch.ones(2,3)                 # all 1
torch.rand(2,3)                 # random values between 0 and 1
torch.full((3,3), 5)            # a 3x3 filled with 5
torch.eye(5)                    # 5x5 identity matrix
torch.arange(0,10,2)            # tensor([0, 2, 4, 6, 8])  -> start, stop, step
torch.linspace(0,10,10)         # 10 evenly spaced values from 0 to 10

torch.empty(2,3)                # does NOT initialize -> shows whatever garbage
                                # is already sitting at that memory location
# _like  -> same SHAPE as an existing tensor x
torch.empty_like(x)
torch.zeros_like(x)
torch.ones_like(x)
torch.rand_like(x)
```

- **requires_grad=True**
	- Tells PyTorch to **track all operations** performed on this tensor
	- During backpropagation PyTorch then calculates the gradient of the loss with respect to this tensor
	- While training a NN we want `dL/dw` and `dL/db` -> so we always create the **weight and bias** tensors with `requires_grad=True`
```python
torch.tensor([[1,2,3],[4,5,6]], requires_grad=True)
```

### DataTypes

| Data Type | Dtype | When it is used |
| --------- | ----- | --------------- |
| 32-bit Floating Point | `torch.float32` | **the standard for deep learning** — best balance of precision and memory |
| 64-bit Floating Point | `torch.float64` | double precision. High-precision numerical tasks, but more memory |
| 16-bit Floating Point | `torch.float16` | half precision. **Mixed-precision training** to cut memory/compute on modern GPUs |
| BFloat16 | `torch.bfloat16` | brain float. Reduced precision vs float16, used in mixed precision especially on TPUs |
| 8-bit Floating Point | `torch.float8` | ultra low precision, experimental / extreme memory constraints |
| 8-bit Integer | `torch.int8` | **quantized models** — save memory and compute at inference |
| 16-bit Integer | `torch.int16` | special numerical tasks needing intermediate precision |
| 32-bit Integer | `torch.int32` | standard signed integer, indexing and general purpose |
| 64-bit Integer | `torch.int64` (long) | large indexing arrays. **Class labels for CrossEntropyLoss must be this** |
| 8-bit Unsigned Integer | `torch.uint8` | **image data** (pixel values 0-255) |
| Boolean | `torch.bool` | masks in logical operations |
| Complex 64 / 128 | `torch.complex64/128` | scientific and signal processing |
| Quantized int/uint 8 | `torch.qint8` / `torch.quint8` | quantized models for efficient inference |

> [!warning] float32 is the default you want
> `torch.tensor(numpy_float_array)` gives you **float64**, but `nn` layers are float32 -> `RuntimeError: expected scalar type Float but found Double`.
> Always pass `dtype=torch.float32` when converting from numpy. float64 is also ~2x slower and buys you nothing in deep learning.

### Basic inspection
```python
x.shape           # torch.Size([2, 3])
x.ndim            # number of dimensions
x.dtype           # torch.float32
x.size()          # same as shape
x.numel()         # total number of elements
x.item()          # python number out of a SINGLE element tensor

x.to(torch.float32)   # change the dtype
```

## Tensor Maths

### Scalar operations
```python
x + 2      # addition
x - 2      # subtraction
x * 2      # multiplication
x / 2      # division
x // 2     # integer division
x % 2      # mod
x ** 2     # power
```

### Element wise operations (2 tensors of the same size)
```python
x + y ,  x - y ,  x * y ,  x / y ,  x % y ,  x ** y

torch.abs(x)      # absolute value -> all elements positive
torch.neg(x)      # all elements negative
torch.round(x)
torch.ceil(x)
torch.floor(x)
torch.clamp(x, min, max)   # squeeze values into a range
```
- `x * y` is **element wise**, NOT matrix multiplication. Matrix multiply is `torch.matmul` / `@`

### Reduction operations
```python
torch.sum(x)            # sum of the WHOLE tensor
torch.sum(x, dim=0)     # column wise sum
torch.sum(x, dim=1)     # row wise sum

torch.mean(x)  /  torch.mean(x, dim=0)  /  torch.mean(x, dim=1)
torch.median(x)
torch.max(x)            # largest value
torch.min(x)            # smallest value
torch.prod(x)           # product of all elements
torch.std(x)            # standard deviation
torch.var(x)            # variance
torch.argmax(x)         # POSITION of the largest element
torch.argmin(x)         # POSITION of the lowest element
```

> [!tip] Remembering `dim`
> `dim=0` -> collapse **down the rows** = you get one value **per column**
> `dim=1` -> collapse **across the columns** = you get one value **per row**
> Rule: `dim` is the axis that **disappears**.
> `torch.argmax(x, dim=1)` is how you turn multi class logits into predicted class labels.

### Matrix operations
```python
torch.matmul(f, g)         # matrix multiplication   (or  f @ g)
torch.dot(vec1, vec2)      # dot product -> 1D tensors ONLY
torch.transpose(f, 0, 1)   # swap dim 0 and dim 1     (or  f.T for 2D)
torch.det(h)               # determinant   -> needs a float dtype
torch.inverse(h)           # inverse       -> needs a float dtype
```

### Comparison operations
```python
i > j  ,  i < j  ,  i == j  ,  i != j  ,  i >= j  ,  i <= j
```
- These return a **boolean tensor**, which is what you use for masking and for counting correct predictions

### Special functions
```python
torch.log(k)
torch.exp(k)
torch.sqrt(k)
torch.sigmoid(k)
torch.softmax(k, dim=0)
torch.relu(k)
```

### In-place operations
```python
m.add_(n)     # m = m + n , modifies m directly
m.relu_()     # applies relu to m directly
```
- The **trailing underscore** = "do it in place, don't return a new tensor"
- Saves memory, **but** autograd can complain if you modify a tensor it still needs for the backward pass

### Copying a tensor
```python
b = a          # a and b point to the SAME memory -> changing one changes the other
b = a.clone()  # a and b are now independent
```

### Reshaping
```python
a.reshape(2,2,2,2)
a.view(2,2,2,2)     # like reshape but only works on contiguous tensors
a.flatten()         # squash everything into 1 D
a.squeeze()         # REMOVE dimensions of size 1   -> (1,5,1) becomes (5,)
a.unsqueeze(0)      # ADD a dimension of size 1 at position 0 -> (5,) becomes (1,5)
```
- `unsqueeze(0)` is what you use to add the **batch dimension** for a single sample before `model(x)`

## NumPy ↔ Tensor
```python
# numpy array  ->  tensor
a = np.array([1,2,3])
b = torch.from_numpy(a)                    # SHARES memory with the numpy array
b = torch.tensor(a, dtype=torch.float32)   # COPIES + lets you set the dtype (safer)

# tensor  ->  numpy array
a = torch.tensor([1,2,3,4])
b = a.numpy()                              # SHARES memory
b = a.detach().cpu().numpy()               # what you actually need if a is on GPU
                                           # and/or part of a computation graph
```
> [!warning] `from_numpy` / `.numpy()` share memory
> They do **not** copy. Editing the numpy array edits the tensor and vice versa. If you want a real copy use `torch.tensor(...)` (which copies) or `.clone()`.
> Also `.numpy()` fails on a GPU tensor or one with `requires_grad=True` -> that is why the full incantation is `.detach().cpu().numpy()`.

## GPU
```python
# is a GPU available
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU not available. Using CPU.")

# the standard one-liner
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.cuda.get_device_name(0)   # the NAME of the gpu, e.g. 'Tesla T4' (a string, for printing)

# create a tensor directly on the gpu
torch.rand((2,3), device=device)

# move an existing cpu tensor to the gpu
new_x = x.to(device)
```
> [!warning] `get_device_name(0)` returns a **string**, not a device
> `device = torch.cuda.get_device_name(0)` then `torch.rand((2,3), device=device)` will fail — you are handing it `'Tesla T4'` where it expects a device object.
> Use `torch.device('cuda')` for the device, and keep `get_device_name` purely for printing which card you got.

## Autograd

- Autograd provides **automatic differentiation** for tensor operations
- When a tensor has `requires_grad=True`, PyTorch starts building a **computation graph** as you operate on it
- `grad_fn` showing up in the print is the proof that a tensor is part of a graph

### The vocabulary
```
Computation Graph      x = torch.tensor(4.0, requires_grad=True)
                       y = x**2          z = torch.sin(y)

   (x) ──> [Power] ──> (y) ──> [Sin] ──> (Z)
    │         │         │        │        │
  Leaf     Operation  Inter-  Operation  Root
  Node       Node     mediate   Node     Node
                       Node
```
- **Leaf node** — a tensor **you** created with `requires_grad=True` (your weights and biases). Only these get a populated `.grad`
- **Operation node** — the function applied (`PowBackward0`, `SinBackward0`)
- **Intermediate node** — computed along the way. By default you **cannot** get `y.grad`
- **Root node** — the end of the graph. You call `.backward()` on this (in real training, the **loss**)

### Example 1 — one operation
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2

x            # tensor(3., requires_grad=True)
y            # tensor(9., grad_fn=<PowBackward0>)

y.backward()      # calculates dy/dx
print(x.grad)     # tensor(6.)     -> dy/dx = 2x = 6
```

### Example 2 — chained operations
```python
x = torch.tensor(4.0, requires_grad=True)
y = x**2
z = torch.sin(y)

x            # tensor(4., requires_grad=True)
y            # tensor(16.,  grad_fn=<PowBackward0>)
z            # tensor(-0.2879, grad_fn=<SinBackward0>)

z.backward()      # calculates dz/dx  (chain rule, all the way back)
print(x.grad)     # tensor(-7.6613)
print(y.grad)     # None + UserWarning -> y is an INTERMEDIATE node
```
- Rule: you call `.backward()` on the **root** node, and you read `.grad` off the **leaf** nodes
- If you genuinely need an intermediate's grad -> call `y.retain_grad()` **before** backward

### Example 3 — a single perceptron (the real thing)
```python
x = torch.tensor(6.7)
y = torch.tensor(0.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

z = w*x + b
y_pred = torch.sigmoid(z)

def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8                                       # to prevent log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target*torch.log(prediction) + (1-target)*torch.log(1-prediction))

loss = binary_cross_entropy_loss(y_pred, y)
loss.backward()

print(w.grad)    # tensor(6.6918)   -> dL/dw
print(b.grad)    # tensor(0.9988)   -> dL/db
```
```
Computation Graph

  (b) ──────────────┐
                    ↓
  (w) ──> [ * ] ──> [ + ] ──> (z) ──> [Sigmoid] ──> (y_pred) ──┐
           ↑                                                    ↓
  (x) ─────┘                                          [Loss function] ──> loss
                                                                ↑
                                              (y) ──────────────┘
```
- `w` and `b` are the **leaves** (that's why they get `requires_grad=True`), `loss` is the **root**
- `x` and `y` are just data -> **no** `requires_grad`
- These examples use scalars, but **exactly the same thing works on tensors of any dimension**

### What `.backward()` actually does
1. **Traverses** the graph from that root tensor back to the leaf tensors
2. **Computes and accumulates** gradients into `.grad`
3. **Frees all the saved intermediate buffers** to save memory — so you **cannot** backpropagate through the same graph again unless you explicitly say otherwise

### Gradients ACCUMULATE
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)      # tensor(4.)      correct: dy/dx = 2x = 4

# now run  y = x**2  and  y.backward()  again
# (note: we do NOT re-run the  x = torch.tensor(...)  line)
print(x.grad)      # tensor(8.)      <- it ADDED to the old value
```
- For each forward pass we do **not** want the gradient to accumulate
- **Solution** -> `x.grad.zero_()`, or in real training `optimizer.zero_grad()`

### Disabling gradient tracking
```python
# approach 1 - flip the flag in place
x.requires_grad_(False)

# approach 2 - get a detached copy
z = x.detach()

# approach 3 - a whole block (this is the one you use in practice)
with torch.no_grad():
    y = x**2
# y has no grad_fn -> y.backward() is not possible
```
- Use it for: the **weight update** step, **inference**, and **evaluation**

### "What if" questions — the graph-freeing trap
Setup for all three:
```python
x = torch.tensor(4.0, requires_grad=True)
y = x**2
z = torch.sin(y)
```

**Q1 — `y.backward()` then `z.backward()`?** ❌ RuntimeError

**Q2 — `z.backward()` then `y.backward()`?** ❌ RuntimeError

**Q3 — `z.backward()` twice on one forward pass?** ❌ RuntimeError

> `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Specify retain_graph=True if you need to backward through the graph a second time.`

- **Why:** building `y` and `z` stores `grad_fn` **plus the intermediate buffers** needed to compute gradients. The first `.backward()` computes its gradients and then **frees those buffers**. The second call follows the graph structure, finds the data gone, and raises.
- **Why `grad_fn` is still there but it still fails:** `grad_fn` is metadata and is not cleared by the backward pass. It only shows the **structure** of your computation, not the live data needed to compute gradients again.
	- Like having a book's **table of contents** but having thrown away all the **pages** — knowing the structure doesn't help you read it again.
- **Solution**, if you genuinely need two backward calls off one forward pass:
```python
y.backward(retain_graph=True)   # keeps the necessary internal buffers
z.backward()                    # okay to call now
```

## The Training Pipeline
> [!abstract] The 6 steps — memorise this order
> 0. Decide the basics — number of **epochs** and **learning rate**
> 1. **Create a model** — a class; `model` is an object of it; the constructor holds the weights and bias
> 2. **Forward pass** — `z = w*x + b`, `y_pred = sigmoid(z)` (a method inside the model class)
> 3. **Calculate loss** — `loss = loss_function(y_pred, y)`
> 4. **Backward pass** — `loss.backward()` — autograd computes `dL/dw` and `dL/db`. This is the **only** line you write
> 5. **Update parameters** — `w_new = w_old - learning_rate * (dL/dw)`, same for `b`
> 6. **Zero the gradients** — so they do not accumulate into the next iteration

## Level 1 — Everything by hand (no nn module)
- Worth doing once, because it shows exactly what `nn.Linear` and the optimizer replace

```python
class MySimpleANN:
    def __init__(self, x):
        self.w = torch.rand(x.shape[1], 1, requires_grad=True, dtype=torch.float32)  # matrix
        self.b = torch.zeros(1, requires_grad=True, dtype=torch.float32)             # vector

    def forward(self, x):
        z = self.b + torch.matmul(x, self.w)
        return torch.sigmoid(z)                    # shape (n_rows, 1)

    def calculate_loss(self, y_pred, y):           # binary cross entropy, by hand
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)   # clamp to avoid log(0)
        return -(y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)).mean()

epochs, learning_rate = 25, 0.03
ann_model = MySimpleANN(x_train_tensor)

for i in range(epochs):
    y_pred = ann_model.forward(x_train_tensor)                  # 2. forward
    loss   = ann_model.calculate_loss(y_pred, y_train_tensor)   # 3. loss
    loss.backward()                                             # 4. backward

    with torch.no_grad():                                       # 5. update
        ann_model.w -= learning_rate * ann_model.w.grad
        ann_model.b -= learning_rate * ann_model.b.grad

    ann_model.w.grad.zero_()                                    # 6. zero
    ann_model.b.grad.zero_()
    print(f"loss at epoch {i+1} = {loss.item()}")
```
```
Computation Graph for the above

 (w) ──Parameter──┐
                  ↓
                [MatMul] ──┐
                  ↑        ↓
 (x_train) ─Input─┘      [Add] ──> [Sigmoid] ──┐
                           ↑                    ↓
 (b) ──────Parameter───────┘                 [BCE Loss] ──loss──> [Backward Pass]
                                                 ↑                   │  grad_w
 (y_train) ─────────────Target───────────────────┘                   │  grad_b
                                                                     ↓
                                                              [Update w and b]
```

> [!warning] 2 classic mistakes here
> **Mistake 1 —** writing `self.w = self.w - lr * grad` instead of `self.w -= lr * grad`
> The first creates a **brand new tensor**, which does not carry the same `.grad` storage or `requires_grad` behaviour expected of a parameter -> from the next iteration autograd is broken and silently wrong. Always use in-place `-=` inside `torch.no_grad()`.
>
> **Mistake 2 —** forgetting `.grad.zero_()`
> PyTorch accumulates. Without clearing, step 5's gradient is the sum of every step before it.

## Level 2 — the nn Module
> [!abstract] What nn buys you
> 1. Building the network out of ready made **layers**
> 2. Built in **activation functions**
> 3. Built in **loss functions**
> 4. Built in **optimizers**

- Your class **must** inherit `nn.Module`, otherwise nothing from nn works
- You **must** call `super().__init__()` to invoke the parent (nn.Module) constructor
- `nn.Linear(in_features, out_features)` — a linear layer. `Linear` is a **class**, `linear_layer_1` is an **object** of it. Weights are initialized from a uniform distribution by default, bias too
	- `out_features=1` -> a layer with a **single perceptron**
- `nn.Sigmoid` etc are **sub classes** of nn, not plain functions -> you instantiate them
- The computation method **must** be named `forward` (PyTorch design) if you want to use `model(input)` and get all the framework features

### 2a — nn layers, but still updating weights by hand
```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, x):
        super().__init__()                    # invoke the nn.Module constructor
        self.linear_layer_1 = nn.Linear(x, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.linear_layer_1(x)            # wx+b
        return self.sigmoid(z)

model = Model(x_train_tensor.shape[1])

for i in range(epochs):
    y_pred = model(x_train_tensor)            # calls forward() internally
    loss = model.calculate_loss(y_pred, y_train_tensor)
    loss.backward()

    with torch.no_grad():
        model.linear_layer_1.weight -= learning_rate * model.linear_layer_1.weight.grad
        model.linear_layer_1.bias   -= learning_rate * model.linear_layer_1.bias.grad

    model.linear_layer_1.weight.grad.zero_()
    model.linear_layer_1.bias.grad.zero_()

print(model.linear_layer_1.weight)
print(model.linear_layer_1.bias)
```

### 2b — + built in loss and optimizer (the real version)
```python
loss_function = nn.BCELoss()                  # BCELoss is a class, loss_function an object
model = Model(x_train_tensor.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(epochs):
    optimizer.zero_grad()                     # 6 (moved to the top — safer)
    y_pred = model(x_train_tensor)            # 2
    loss = loss_function(y_pred, y_train_tensor)   # 3
    loss.backward()                           # 4
    optimizer.step()                          # 5  -> replaces the whole no_grad block
    print(f"loss at epoch {i+1} = {loss.item()}")
```
- `optimizer.step()` replaces the manual `w -= lr * w.grad` for **every** parameter at once
- `optimizer.zero_grad()` replaces every manual `.grad.zero_()`

### 2c — multi layer, without Sequential
- input layer (5 features, no perceptron here) -> hidden layer (3 perceptrons) -> output layer (1 perceptron)
```python
class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.linear_layer_1 = nn.Linear(x, 3)
        self.linear_layer_2 = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.linear_layer_1(x)     # wx+b
        a1 = self.relu(z1)
        z2 = self.linear_layer_2(a1)
        return self.sigmoid(z2)

features = torch.rand(10,5)             # toy x_train
model = Model(features.shape[1])
model(features)                         # forward pass
```
- The `out_features` of one layer **must** equal the `in_features` of the next (`3` -> `3`)

### 2d — multi layer, with a Sequential container
```python
class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
```
- Same model, far less typing. Use Sequential whenever the data just flows **straight through** top to bottom
- Go back to writing `forward` by hand when you need branches, skip connections or multiple inputs

### Visualizing the model
```python
print(model.linear_layer_1.weight)
print(model.linear_layer_1.bias)
model.parameters()      # generator over all trainable tensors -> feed to the optimizer
model.state_dict()      # dict of all weights

!pip install torchinfo
from torchinfo import summary
summary(model, input_size=(10,5))       # (batch_size, n_features)
```
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [10, 1]                   --
├─Sequential: 1-1                        [10, 1]                   --
│    └─Linear: 2-1                       [10, 3]                   18     # 5*3 + 3
│    └─ReLU: 2-2                         [10, 3]                   --
│    └─Linear: 2-3                       [10, 1]                   4      # 3*1 + 1
│    └─Sigmoid: 2-4                      [10, 1]                   --
==========================================================================================
Total params: 22
```

## Activation + Loss pairing
> [!abstract] The core choice
> You have exactly **2 valid options**:
> 1. **Remove the last activation** and use a loss that applies the activation internally ✅ recommended
> 2. **Keep the last activation** and use a loss that does **not** apply it
> Doing both = applying the activation **twice** = the model barely learns. Doing neither = wrong maths.

### Regression
| Output | Setup |
| ------ | ----- |
| Continuous value `[1.123, 8.945]` | **no activation + `nn.MSELoss()`** ✅ |
| Non continuous value `[0,1,2]` | `nn.Sigmoid()` + `nn.MSELoss()` , or no activation + `nn.MSELoss()` |

### Binary classification
| Setup | Note |
| ----- | ---- |
| **no final activation (output = logit) + `nn.BCEWithLogitsLoss()`** ✅ | computes the sigmoid internally in a **numerically stable** way |
| `nn.Sigmoid()` final layer + `nn.BCELoss()` | less preferred — a separate Sigmoid → BCELoss is less numerically stable and can under/overflow |

### Multi class classification
| Setup | Note |
| ----- | ---- |
| **no final activation (raw logits, shape `[N, C]`) + `nn.CrossEntropyLoss()`** ✅ | expects logits, internally applies `log_softmax` + `NLLLoss` in a numerically stable way |
| `nn.LogSoftmax(dim=1)` final layer + `nn.NLLLoss()` | valid. **Do NOT use plain Softmax + NLLLoss** — NLLLoss expects **log**-probabilities |

- Target shapes: `BCEWithLogitsLoss` wants **float**, same shape as the output `(N,1)`. `CrossEntropyLoss` wants a **class index**, `dtype=torch.long`, shape `(N,)` — **not** one-hot
- Imbalanced data -> `pos_weight=` on BCEWithLogitsLoss, `weight=` on CrossEntropyLoss

## Optimizers
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          # sensible default
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
# call scheduler.step(val_loss) at the end of each EPOCH (not each batch)
```
- **lr** — the single most important knob
- **weight_decay** — this **is** L2 regularization

## Dataset & DataLoader
- Dataset and DataLoader are core abstractions that **decouple how you define your data from how you efficiently iterate over it** in training loops

```
        The Dataset  (a database, a csv, a folder of images)
                   ▲
                   │
        ┌──────────────────┐          ┌──────────────────┐
        │  Dataset Class   │ ◀─────── │ DataLoader Class │
        │   __len__        │          └──────────────────┘
        │   __getitem__    │
        └──────────────────┘
```

### Dataset class
- Responsible for **loading the data from memory**. It is an **abstract class** in PyTorch -> you inherit it and define:

| Method | What it does |
| ------ | ------------ |
| `__init__()` | tells how the data should be loaded |
| `__len__()` | returns the **total number of samples** |
| `__getitem__(index)` | returns the data (and label) at the given index. **Apply all your data transformations here** |

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features_x_array, labels_y_array):
        self.features_x_array = features_x_array
        self.labels_y_array = labels_y_array

    def __len__(self):
        return self.features_x_array.shape[0]

    def __getitem__(self, index):
        return self.features_x_array[index], self.labels_y_array[index]

dataset = CustomDataset(X, y)

# now len() and [] work on the object
dataset[1]     # (tensor([-2.8954,  1.9769]), tensor(0))
len(dataset)   # 10
```

### DataLoader class
- **The iterator you put in your training loop.** Responsible for:
	1. Deciding **which indices** to fetch (via a **Sampler**)
	2. Spawning **worker** processes/threads to call your `__getitem__` in parallel
	3. **Batching** the single-sample outputs into multi-sample tensors (via a **BatchSampler** + **collate_fn**)

```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_features, batch_labels in dataloader:
    print(batch_features)
    print(batch_labels)
    print("-"*50)
```

### What actually happens, step by step
Your CSV has 10 rows -> indices 0-9. You set `batch_size=2, shuffle=True, num_workers=2`.

**1. Epoch starts — the Sampler shuffles the indices**
```
shuffled_indices = [9, 2, 5, 0, 6, 4, 7, 1, 3, 8]
```
- Up to this point **no data has been loaded from disk** — you have only created a new ordering of integers

**2. BatchSampler chunks them into batches of 2**
```
batch_indices_list = [ [9,2], [5,0], [6,4], [7,1], [3,8] ]
```
- Still **nothing fetched**. This just says "we will load samples 9 and 2 as batch 1, then 5 and 0 as batch 2, ..."

**3. Fetching & collating — now the actual loading begins**
- For the first batch `[9, 2]`:
	- Worker 1 calls `dataset.__getitem__(9)` -> reads row 9 -> returns `(x9, y9)`
	- Worker 2 calls `dataset.__getitem__(2)` -> reads row 2 -> returns `(x2, y2)`
	- the **collate function** stacks these into tensors of shape `(2, ...)`
	- that stacked tensor becomes **Batch 1**
- Repeat for each sub-list

- **What is the collate function** — it specifies **how to combine a list of samples into a single batch**. By default DataLoader uses a simple stacking mechanism (`default_collate`), but `collate_fn` lets you customise how the data is processed and batched (essential for variable-length sequences / NLP padding)

### Samplers
- **Input:** the dataset. **Output:** a sequence of indices like `[3,8,9,0,...]`

| Sampler | What it does | Use for |
| ------- | ------------ | ------- |
| **SequentialSampler** | returns indices in order 0..N-1, every epoch | validation / testing, **time series** data — anywhere you don't want randomness |
| **RandomSampler** | a new random permutation of `[0..N-1]` each epoch | **training** — mix up examples to reduce over-fitting |
| **WeightedRandomSampler** | samples **with replacement**, each index i has its own weight wᵢ | **class imbalance** — give under-represented classes higher weights so they appear more often |
| Others | `SubsetRandomSampler`, `BatchSampler`, `DistributedSampler` | subsets, custom batching, multi-GPU |

### DataLoader parameters
| Parameter | Meaning | Values |
| --------- | ------- | ------ |
| **dataset** | the Dataset class object | — |
| **batch_size** | how many samples to group into one batch before handing it to your training loop | int, e.g. `32` |
| **shuffle** | if True, randomly reorders the indices **each epoch** so batches come in a new order | True / False |
| **num_workers** | how many background helper processes load samples in parallel | int, e.g. `0`, `4` |
| **pin_memory** | if True, puts loaded tensors in pinned (faster) CPU RAM so transfers to GPU are quicker | True / False |
| **drop_last** | if True, drops the final batch when it has fewer than `batch_size` samples | True / False |
| **sampler** | if you don't pass one and set `shuffle=True`, PyTorch uses a `RandomSampler`. You can pass an existing one or write your own | — |
| **collate_fn** | default is `default_collate`. You can pass a different one or write your own | — |

- **`drop_last=True` matters when you use BatchNorm** — a final batch of size 1 makes `BatchNorm1d` throw (it cannot compute a variance over a single sample)
- **`shuffle=True` reshuffles every epoch automatically** -> do **not** recreate the DataLoader inside the epoch loop
- **`shuffle=False` on test/eval** so evaluation is deterministic
- batch_size `1` = Stochastic GD, `len(dataset)` = Batch GD, in between = **Mini Batch GD**
- If x and y are already tensors and you need no transforms -> skip the custom class, use `TensorDataset(x, y)`

## The Training Loop
```python
model = MyAnnModel(x_train_tensor.shape[1])
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    model.train()
    # DataLoader(..., shuffle=True) reshuffles every epoch by itself.
    # You do NOT need to re-create it inside the loop.
    for (x, y) in train_dataLoader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"epoch {i} - loss = {loss.item()}")
```
- Use `loss.item()` when printing/accumulating — otherwise you keep the whole graph alive in memory and leak

### `model.train()` vs `model.eval()`
- Both toggle a module-level flag (`module.training = True/False`) that changes the behaviour of certain layers
- **Layers that change behaviour:** `nn.Dropout` / `nn.Dropout2d`, all `nn.BatchNorm*`, and some other modules

| You forget | What happens |
| ---------- | ------------ |
| **`model.train()` during training** | dropout is **disabled** and BatchNorm uses **running stats** — both reduce regularization and change the optimization dynamics. You are effectively training with **inference behaviour**: BN running stats won't update properly and dropout's effect is missing |
| **`model.eval()` during validation** | measurements are **noisy and biased** — dropout randomly zeros units (lower accuracy) and BatchNorm updates its running stats and uses batch stats. Results become unreliable **and you accidentally mutate model state** (BN buffers), polluting what you thought was a pure validation step |

> [!warning] `model.eval()` and `torch.no_grad()` are different things
> `model.eval()` changes **layer behaviour**. `torch.no_grad()` stops **graph building**.
> You need **both** for evaluation, and `model.train()` again at the top of the next epoch.

## Evaluation
```python
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_dataLoader:
        logits = model(xb)

        # binary (output shape (n,1), model ends in raw logits)
        prediction = (torch.sigmoid(logits) > 0.5).float()

        # multi class (output shape (n, n_classes))
        # _, prediction = torch.max(logits, 1)

        total   += yb.shape[0]
        correct += (prediction == yb).sum().item()
print("accuracy:", correct/total)
```
> [!warning] `torch.max(logits, 1)` is for MULTI CLASS only
> With a single output neuron of shape `(n,1)`, `torch.max(..., 1)` always returns index 0 -> your accuracy comes out silently wrong. For binary, threshold the sigmoid instead.

## Training on GPU — the exact changes
1. **Check availability**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```
2. **Add `pin_memory` to the DataLoaders**
```python
train_dataLoader = DataLoader(train_dataset_object, batch_size=10, shuffle=True,  pin_memory=True)
test_dataLoader  = DataLoader(test_dataset_object,  batch_size=10, shuffle=False, pin_memory=True)
```
3. **Move the model**
```python
model = model.to(device)
```
4. **Move each batch, inside both the training and the evaluation loop**
```python
x, y = x.to(device), y.to(device)
```
- Model and data must be on the **same** device -> otherwise `RuntimeError: Expected all tensors to be on the same device`
- To bring a tensor back for numpy/printing -> `t.detach().cpu().numpy()`

## Optimizing the Neural Network
> [!abstract] Placement rule
> **BatchNorm** -> applied to hidden layers **BEFORE** the activation function
> **Dropout** -> applied to hidden layers **AFTER** the activation function
> So the order inside a block is: `Linear -> BatchNorm -> Activation -> Dropout`

```python
class MyAnnModel(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(number_of_features, 300),
            nn.BatchNorm1d(300),      # = number of outputs from the previous layer
            nn.ReLU(),
            nn.Dropout(p=0.3),        # drop 30% of neurons while training

            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(50, 1)          # raw logit -> BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.network(x)
```
- `nn.BatchNorm1d(n)` -> `n` must equal the **out_features of the layer above it**
- **weight decay == L2 regularization**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
```
- Dropout and BatchNorm only behave correctly if you call `model.train()` / `model.eval()` properly

## Hyper parameter tuning with Optuna
- Values worth searching over:
	- Number of **epochs**
	- Number of **layers**
	- Number of **neurons** in each layer
	- Which **optimizer**
	- **Batch size**
	- **Learning rate**
	- **Dropout rate**
	- **Weight decay** [lambda]

```python
import optuna

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units  = trial.suggest_categorical('n_units', [50, 100, 300])
    dropout  = trial.suggest_float('dropout', 0.1, 0.5)
    lr       = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    opt_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    # ...build the model with these, train it, return the validation metric
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
```
- `log=True` on the learning rate matters — lr should be searched on a **log scale**, not linear

## Save & Load
```python
# recommended -> save only the weights
torch.save(model.state_dict(), 'model.pth')

model = MyAnnModel(n_features)          # rebuild the same architecture first
model.load_state_dict(torch.load('model.pth'))
model.eval()

# checkpoint (to resume training later)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

## Reproducibility
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
```

## Full Pipeline (end to end template)
```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. data
np.random.seed(42)
X = np.random.rand(200, 30)
Y = np.random.randint(0, 2, size=(200, 1))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 2. tensors  (float32 !)
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. dataset + dataloader
class MyCustomDataSet(Dataset):
    def __init__(self, x_array, y_array):
        self.x_array, self.y_array = x_array, y_array
    def __len__(self):
        return self.x_array.shape[0]
    def __getitem__(self, index):
        return self.x_array[index], self.y_array[index]

train_dataLoader = DataLoader(MyCustomDataSet(x_train_tensor, y_train_tensor),
                              batch_size=10, shuffle=True,  pin_memory=True, drop_last=True)
test_dataLoader  = DataLoader(MyCustomDataSet(x_test_tensor,  y_test_tensor),
                              batch_size=10, shuffle=False, pin_memory=True)

# 4. model
class MyAnnModel(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(number_of_features, 300), nn.BatchNorm1d(300), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(300, 100),                nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(100, 50),                 nn.BatchNorm1d(50),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(50, 1)                    # raw logit
        )
    def forward(self, x):
        return self.network(x)

epochs, learning_rate = 25, 0.001
model = MyAnnModel(x_train_tensor.shape[1]).to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 5. train
for i in range(epochs):
    model.train()
    epoch_loss = 0
    for x, y in train_dataLoader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(x), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"epoch {i+1} - avg loss {epoch_loss/len(train_dataLoader)}")

# 6. evaluate
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_dataLoader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        total   += y.shape[0]
        correct += (pred == y).sum().item()
print("accuracy:", correct/total)
```

## PyTorch vs sklearn mental map

| sklearn | PyTorch |
| ------- | ------- |
| `model = LogisticRegression()` | define an `nn.Module` class |
| `model.fit(X,y)` | you write the epoch/batch loop yourself |
| internal solver | `torch.optim.*` + `loss.backward()` |
| `model.predict(X)` | `model.eval()` + `with torch.no_grad(): model(X)` |
| `GridSearchCV` | Optuna |
| — | you handle device (cpu/gpu), dtype and batching manually |

## Related
- [[CheatSheet-Tensorflow]] — the same models written in Keras
- [[CheatSheet of Machine Learning]] — classical ML
- [[Gradient Decent]] — the optimization theory behind `optimizer.step()`
