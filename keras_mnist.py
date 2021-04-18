# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

# Describe Keras architecture
inputs = keras.Input(shape=(300,), name="digits")
x = layers.Dense(28, activation="tanh", name="dense_1")(inputs)
x = layers.Dense(28, activation="tanh", name="dense_2")(x)
outputs = layers.Dense(10, activation="sigmoid", name="predictions")(x)

# Keras out-of-the-box activations 
'''
 relu function
sigmoid function
softmax function
softplus function
softsign function
tanh function
selu function
elu function
exponential function
'''
# create a model with the given architecture 
model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(maxlen=(300))
# Fill with zeros the empty places
x_train = [np.resize(np.array(t),300) for t in x_train]
x_test = [np.resize(np.array(t),300) for t in x_test]

x_train = np.array(x_train)
x_test = np.array(x_test)
# Preprocess the data (these are NumPy arrays)
#x_train = x_train.reshape(60000, 784).astype("float32") / 255
#x_test = x_test.reshape(10000, 784).astype("float32") / 255

#y_train = y_train.astype("float32")
#y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=8,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)

