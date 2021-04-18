import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import matplotlib.pyplot as plt

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=1000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

def vectorize(sequences, dimension = 1000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  results[i, sequence] = 1
 return results

data_2 = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data_2[:10000]
test_y = targets[:10000]
train_x = data_2[10000:]
train_y = targets[10000:]

model = models.Sequential()
# Input - Layer
model.add(layers.Dense(80, activation = "relu", input_shape=(1000, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(60, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
# compiling the model
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 train_x, train_y,
 epochs= 5,
 batch_size = 50,
 validation_data = (test_x, test_y)
)

print(results.history.keys())

plt.plot(results.history['accuracy'])
plt.plot(results.history['loss'])
plt.plot(results.history['val_accuracy'])
plt.plot(results.history['val_loss'])
plt.legend(['acc', 'loss','val_accuracy','val_loss'], loc='upper left')
plt.show()
