from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print()
print("train_labels:")
print(f"Clase:{type(train_labels)}")
print(f"Dimensiones: {train_labels.ndim}")
print(f"Shape: {train_labels.shape}")
print(f"Clase de los items: {train_labels.dtype}")
    
print()
print("train_images:")
print(f"Clase:{type(train_images)}")
print(f"Dimensiones: {train_images.ndim}")
print(f"Shape: {train_images.shape}")
print(f"Clase de los items: {train_images.dtype}")

print()
print(f"Primeros 5 valores: {train_labels[:5]}")
for n in range(5):
    print(n, train_labels[n])
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()

print()
print("Fin del programa.")
