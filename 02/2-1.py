# 2-1.py
# Clasificador de dígitos escritos a mano MNIST, implementado en Keras
# Carlos Garcia, Zaragoza Diciembre 2025

from keras import models
from keras import Input
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

print()
print("1. Carga de datos")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print()
print("2. Deficición del modelo y preparación de los datos")
# Sequential model, the simplest type of Keras model
network = models.Sequential()

# input_shape=[28 * 28]: Specifies the input is a 784 values (1D array)
network.add(Input(shape=(28 * 28,)))

# Core of the model: 2 Dense layers

# units=512: This specifies there are 512 neurons in this layer
# Activation function for the neuron: 'relu'
network.add(layers.Dense(512, activation='relu'))

# units=10: This specifies there are 10 neurons in this layer
# Activation function for the neurons: 'softmax'
# As this is the last layer, it will be the output layer
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# De array 3D (60000, 28, 28) a 2D (60000, 28*28). 28*28=786 es el número de neuronas en la capa de entrada.
# Typecasting de entero 8 bits a punto flotante 32 bits. Es un requerimiento del algoritmo de backpropagation
# Normalización de rango 0-255 a 0-1. Ayuda a que el modelo 'converja' antes
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32')/255
test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255

# Training
print()
print("3. Model training")
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluación
print()
print("4. Evaluación con datos de test que el modelo no ha visto")
test_loss, test_acc = network.evaluate(test_images, test_labels)

print("======================================================")
print(f"Loss con datos de test: {test_loss}")
accu = test_acc * 100
print(f"Accuracy con datos de test: {test_acc} - {accu:.1f}%")

# Fin
print()
print("Fin del programa.")