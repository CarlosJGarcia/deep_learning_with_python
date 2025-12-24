from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print()
print("train_labels:")
print(f"Clase:{type(train_labels)}")
print(f"Dimensiones: {train_labels.ndim}")
print(f"Shape: {train_labels.shape}")
print(f"Clase de los items: {train_labels.dtype}")
"""
print(f"Primeros 5 valores: {train_labels[:5]}")
for n in range(5):
    print(n, train_labels[n])
"""
    
print()
print("train_images:")
print(f"Clase:{type(train_images)}")
print(f"Dimensiones: {train_images.ndim}")
print(f"Shape: {train_images.shape}")
print(f"Clase de los items: {train_images.dtype}")

print("Fin del programa.")
print()