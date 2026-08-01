---
tags:
  - deepLearning
  - tensorflow
  - keras
  - cheatSheet
---
## Imports
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeUniform, HeNormal
```

> [!warning] Keras 3 changes (TF 2.16+)
> - `input_dim=` / `input_shape=` on the first layer is legacy. Use an explicit `Input(shape=(n,))` as the first layer instead — it still works but warns.
> - `Embedding(..., input_length=max_words)` -> `input_length` is **removed**, just drop it.
> - `optimizer.lr=` -> renamed to `learning_rate=`.
> - `keras.preprocessing.image.ImageDataGenerator` is **deprecated** -> use `keras.layers` augmentation layers or `image_dataset_from_directory`.
> - `model.save('m.h5')` -> use the new `model.save('m.keras')` format.

## ANN / Multi Layer Perceptron

### The full stack (optimizer + init + regularization + batchnorm + dropout)
```python
model = Sequential()
adam = Adam(learning_rate=0.01)

model.add(Dense(11, activation='elu', input_dim=11,
                kernel_initializer=HeUniform(),
                kernel_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(11, activation='elu',
                kernel_initializer=HeUniform(),
                kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
```

- **Weight initialization** -> `GlorotUniform` is the **default**
	- Glorot/Xavier -> use with **sigmoid / tanh**
	- He -> use with **ReLU and its family** (relu, elu, leaky relu)
- **Regularization**
	- Applying **L2 in all hidden layers is a good default strategy**
	- Selectively applying **L1** in some layers helps when a **sparse** feature representation is valuable
- **BatchNormalization**
	- Comes **above the dropout layer** (Dense -> BatchNorm -> Dropout)
	- Only used with **mini batch** gradient descent
- **Activations - the dying ReLU problem**
	- **ELU** and **SELU** are excellent alternatives to ReLU
	- ELU converges faster in some cases due to its smoother output
	- SELU enables **self normalization** in deep networks -> can replace the need for batch normalization in specific architectures

### Output layer + compile (this is the part people get wrong)

| Problem | Last layer | Loss | metrics |
| ------- | ---------- | ---- | ------- |
| **Binary classification** | `Dense(1, activation='sigmoid')` | `binary_crossentropy` | `['accuracy','AUC']` |
| **Multi class** (integer labels) | `Dense(10, activation='softmax')` | `sparse_categorical_crossentropy` | `['accuracy']` |
| **Multi class** (one-hot labels) | `Dense(10, activation='softmax')` | `categorical_crossentropy` | `['accuracy']` |
| **Regression** | `Dense(1, activation='linear')` | `mean_squared_error` | `['mse','mae']` |

```python
# binary classification
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy','AUC'])

# multi class classification
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# regression
model.add(Dense(1, activation='linear'))
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse','mae'])
```

> [!tip] sparse vs non-sparse
> `sparse_categorical_crossentropy` -> y is an **integer** (`[0,2,1,...]`)
> `categorical_crossentropy` -> y is **one-hot** (`[[1,0,0],[0,0,1],...]`, i.e. after `to_categorical`)
> Picking the wrong one is the #1 shape error in Keras.

### Metrics you can pass to `metrics=`
- **Binary + multi class:** `accuracy`, `AUC`, `TruePositives`, `TrueNegatives`, `FalsePositives`, `FalseNegatives`, `PrecisionAtRecall`, `RecallAtPrecision`, `Precision`, `Recall`
- **Multi class only:** `categorical_accuracy` (targets one-hot), `sparse_categorical_accuracy` (targets integer), `categorical_hinge`
- **Regression:** `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`

### EarlyStopping
```python
callback = EarlyStopping(
    monitor="val_loss",          # watch validation loss to decide when to stop
    min_delta=0.00001,           # minimum improvement to count as improvement
    patience=3,                  # stop after 3 epochs with no improvement
    verbose=1,                   # print a message when stopping
    mode="auto",                 # auto infer the direction (min for val_loss)
    restore_best_weights=True    # roll back to the best epoch's weights
)
```
> [!warning] `restore_best_weights=False` is the default and it is almost always wrong
> With `False` you keep the weights from the **last** (worse) epoch — the 3 epochs of degradation that triggered the stop. Set it to **True** so you keep the best model.

### Fit
```python
history = model.fit(X_train, y_train,
                    batch_size=50, epochs=100, verbose=1,
                    validation_split=0.2, callbacks=[callback])

# or with an explicit validation set
history = model.fit(X_train, y_train,
                    batch_size=50, epochs=100, verbose=1,
                    validation_data=(X_val, y_val), callbacks=[callback])
```
- **batch_size = 50** -> Mini Batch Gradient Descent
- **batch_size = 1** -> Stochastic Gradient Descent
- **batch_size = len(X_train)** -> Batch Gradient Descent
- `callbacks=` expects a **list**
- `validation_split=0.2` takes the **last 20% of rows as-is, without shuffling** -> shuffle your data before, or use `validation_data=`

### Predict
```python
y_prob = model.predict(X_test)

# binary -> threshold the sigmoid probability
y_pred = np.where(y_prob > 0.5, 1, 0)

# multi class -> pick the highest softmax probability
y_pred = np.argmax(y_prob, axis=1)
```

### Plotting the history
```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
# train loss falling while val loss rises = over-fitting
```

## Functional API
- Use it when Sequential is not enough -> **multiple inputs, multiple outputs, branches, skip connections**

### Multiple output with tabular input
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

input_layer = Input(shape=(3,))
dense1 = Dense(128, activation='relu')(input_layer)
dense2 = Dense(64,  activation='relu')(dense1)

# Branch 1 -> classification
branch1 = Dense(12, activation='relu')(dense2)
output1 = Dense(1, activation='sigmoid', name='Classification_Output')(branch1)

# Branch 2 -> regression
branch2 = Dense(12, activation='relu')(dense2)
output2 = Dense(1, activation='linear', name='Regression_Output')(branch2)

model = Model(inputs=input_layer, outputs=[output1, output2])

model.compile(optimizer=Adam(),
              loss={'Classification_Output':'binary_crossentropy',
                    'Regression_Output':'mse'},
              metrics={'Classification_Output':'accuracy',
                       'Regression_Output':'mse'})

history = model.fit(
    X_train,
    {"Classification_Output": y_train_classification,
     "Regression_Output":     y_train_regression},
    epochs=10, batch_size=32, validation_split=0.2
)
```
- **`name=` on the output layers is mandatory here** — it is the key you use in the loss/metrics/target dicts

### Multiple input
```python
input_layer1 = Input(shape=(5,))
input_layer2 = Input(shape=(5,))

dense1_1 = Dense(128, activation='relu')(input_layer1)
dense1_2 = Dense(64,  activation='relu')(dense1_1)

dense2_1 = Dense(128, activation='relu')(input_layer2)
dense2_2 = Dense(64,  activation='relu')(dense2_1)

concat = Concatenate()([dense1_2, dense2_2])
dense3 = Dense(8, activation='relu')(concat)
output = Dense(1, activation='linear')(dense3)

model = Model(inputs=[input_layer1, input_layer2], outputs=output)
```

### See the architecture
```python
model.summary()
from keras.utils import plot_model
plot_model(model, show_shapes=True)
```

## RNN
- Many to one RNN (including bidirectional)
```python
from tensorflow.keras.layers import Embedding, SimpleRNN, Bidirectional, Dense

model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(Bidirectional(SimpleRNN(64, return_sequences=True)))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16, return_sequences=True))
model.add(SimpleRNN(4))                      # last RNN -> return_sequences=False
model.add(Dense(1, activation='sigmoid'))
```
> [!abstract] `return_sequences` rule
> **True** -> outputs the hidden state at every timestep, shape `(batch, timesteps, units)`. Needed when the **next layer is another RNN**.
> **False** (default) -> outputs only the last timestep, shape `(batch, units)`. Use on the **last** RNN layer before a Dense.
> Stacking RNNs = every layer except the last one needs `return_sequences=True`.

- `Bidirectional(...)` doubles the output units (forward + backward concatenated)
- SimpleRNN suffers from **vanishing gradient** on long sequences -> that is why LSTM/GRU exist

### LSTM (stacked + bidirectional)
```python
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=100)
```

### GRU (stacked + bidirectional)
```python
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32))
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(GRU(32, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(32))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- **LSTM vs GRU** -> GRU has fewer gates (2 vs 3) = fewer params, trains faster, usually similar accuracy. LSTM tends to win on very long sequences.
- `Embedding(vocab_size, 32)` -> `32` = the size of each word vector

## CNN

### Basic CNN
> [!abstract] The pattern
> As you move forward the **number of filters generally increases** (32 -> 64 -> 128), while **kernel size and pool size stay the same** (3x3 and 2x2).
> Conv -> BatchNorm -> Pool, repeated. Then `Flatten()` and the ANN part, where you can add Dropout.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- `input_shape=(256,256,3)` -> `256x256` image, `3` = coloured (RGB). `1` = grayscale.

### Padding & Strides
- **padding='valid'** -> no padding, the image shrinks after every conv
- **padding='same'** -> pads with zeros so the output keeps the **same** height/width
- **strides** -> how many pixels the filter jumps. Not used much in conv layers, standard in pooling.
```python
model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(2,2),
                 activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
```

### Loading images with generators
- The idea -> load **small batches into RAM** instead of the whole dataset
- Folder structure must be `train/class_a/*.jpg`, `train/class_b/*.jpg` -> then `labels='inferred'` picks the class from the folder name
```python
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',        # 'int' -> sparse_categorical_crossentropy
    batch_size=32,           # 'binary' -> binary_crossentropy
    image_size=(256,256)     # 'categorical' -> categorical_crossentropy
)
validation_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels='inferred', label_mode='int',
    batch_size=32, image_size=(256,256)
)

# Normalize to 0-1
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

### Data Augmentation
- Only ever apply it to the **training** set, never validation/test
```python
# modern way (Keras 3) - augmentation as layers, runs on GPU
data_augmentation = Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomTranslation(0.2, 0.2),
])
model.add(data_augmentation)   # put it as the FIRST layer of the model

# legacy way (deprecated, still seen everywhere)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
datagen.fit(x_train)
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=50,
          validation_data=(x_val, y_val))
```

### Predicting on a single random image
```python
import cv2
test_img = cv2.imread('/content/cat.jpg')
plt.imshow(test_img)

test_img = cv2.resize(test_img, (256,256))         # must match input_shape
test_input = test_img.reshape((1,256,256,3)) / 255.0   # batch dim + same normalization
model.predict(test_input)
```
> [!warning] Two things that silently break single image prediction
> 1. **You must resize** — `reshape` does not resize, it just reinterprets the buffer and will throw if the pixel count differs.
> 2. **Apply the exact same normalization you used in training** (`/255.`). Forgetting this gives garbage predictions with no error.
> Also, `cv2.imread` returns **BGR**, keras models are trained on **RGB** -> `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

## Pretrained Models
- Use the model as-is **only if it was pretrained on your current target task**
```python
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img = keras.utils.load_img('elephant.jpg', target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)     # add the batch dimension
x = preprocess_input(x)           # each model has its OWN preprocess_input, use it

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# [(u'n02504013', u'Indian_elephant', 0.826), (u'n01871265', u'tusker', 0.112), ...]
```

## Transfer Learning

### Feature Extraction
- **Freeze** the weights of the pretrained model and only train the final layers specific to the new task
- Use when your dataset is **small** and similar to the original one
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False        # FREEZE

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```
- **include_top=False** -> do not import the fully connected layers (they are task specific)
- **input_shape=(150,150,3)** -> image size 150, `3` = coloured image

### Fine Tuning
- **Unfreeze** some or all layers of the pretrained model and train them jointly with the new layers
- Use when your dataset is **bigger** or fairly different from the original one
```python
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = True

# unfreeze from 'block5_conv1' onwards, keep everything before it frozen
Flag = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        Flag = True
    layer.trainable = Flag

for layer in conv_base.layers:
    print(layer.name, layer.trainable)

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),   # VERY small lr
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```
> [!warning] Fine tuning needs a tiny learning rate
> `1e-5` here is not a typo. With a normal lr the first few large gradient updates **destroy** the pretrained weights before the new head has learned anything — the whole point of transfer learning is lost.
> Standard practice: do Feature Extraction first (frozen base, normal lr), **then** unfreeze and fine tune at `1e-5`.

## Save & Load
```python
model.save('model.keras')                 # new format (Keras 3)
model = keras.models.load_model('model.keras')

model.save_weights('weights.weights.h5')  # weights only
model.load_weights('weights.weights.h5')
```

## Quick reference - what to reach for

| You have | Use |
| -------- | --- |
| Tabular data, simple stack of layers | `Sequential` + `Dense` |
| Multiple inputs / outputs / branches | **Functional API** |
| Images | `Conv2D` + `MaxPooling2D` -> `Flatten` -> `Dense` |
| Small image dataset | **Transfer learning** (feature extraction) |
| Text / sequences / time series | `Embedding` -> `LSTM` / `GRU` |
| Over-fitting | Dropout, L2, EarlyStopping, data augmentation |
| Training unstable / slow | BatchNormalization, lower lr, He init |
