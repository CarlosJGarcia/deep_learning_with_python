# Modelo Deep Learning que determina si una crítica cinematográfica es positiva o negativa
# Dataset: keras.IMDB
# Ejecución de diferentes números de epochs con cálculo de accuracy en datos que el modelo no ha visto
# Mejor número de epochs: 4

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.datasets import imdb
import matplotlib.pyplot as plt

# Lista vacía de resultados
lista_epochs = []
lista_resultados = []


print("Cargando keras.IMDB")
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Hecho")

# Función de vectorización de datos
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Vectorizar los datos
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorizar las etiquetas
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

def modelo(input_epochs):

    # Definición del modelo
    # Sequential model, the simplest type of Keras model
    model = models.Sequential()

    # Input_shape=10000: Especifica que cada muestra tiene 10000 características (1D array)
    model.add(layers.Input(shape=(10000,)))

    # Core of the model: 3 Dense layers

    # units=16: This specifies there are 16 neurons in this layer
    # Activation function for the neuron: 'relu'
    model.add(layers.Dense(16, activation='relu'))

    # units=16: This specifies there are 16 neurons in this layer
    # Activation function for the neuron: 'relu'
    model.add(layers.Dense(16, activation='relu'))

    # units=1: This specifies there is 1 neuros in this layer
    # Activation function for the neuron: 'sigmoid'
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Entreno el modelo con input_epochs ya que hemos visto que a partir de la 5 pierde precisión con datos que no ha visto
    print("Entrenamiento del modelo")
    history = model.fit(x_train, y_train, epochs=input_epochs, batch_size=512)

    # Evaluo la exactitud (accuracy) del modelo con los datos de Prueba
    results = model.evaluate(x_test, y_test)
    output_accuracy = results[1]*100

    return output_accuracy


# Bucle por el rango de epochs
for n in range (1,61):
    valor_modelo = modelo(n)
    
    lista_epochs.append(n)
    lista_resultados.append(valor_modelo)

# Mostrar el resultado
for n in range (0,60):
    print (f"Epochs: {lista_epochs[n]} -> accuracy: {lista_resultados[n]}%")

# loss_values = history_dict['loss']
# acc = history_dict['accuracy']
# epochs = range(1, len(loss_values)+1)

plt.plot(lista_epochs, lista_resultados, color='blue', label='Evaluated accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

