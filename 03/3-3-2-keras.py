from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(32, activation='relu'))

print("Fin del programa.")