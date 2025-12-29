import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist

model = models.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(32, activation='relu'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(input_tensor, label_tensor), (test_tensor, testlbl_tensor) = mnist.load_data()
input_tensor = input_tensor.reshape(60000, 28*28)
input_tensor = input_tensor.astype('float32')/255

model.fit(input_tensor, label_tensor, batch_size=128, epochs=10)

print("Fin del programa.")