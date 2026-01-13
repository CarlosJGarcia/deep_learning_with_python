# TensorFlow GPU performance test
# Xeon 32 GB RAM + RTX 3060 12 GB: 2' 28''
# M2   16 GB RAM                 : 5' 36''
# i9   32 GB RAM                 :

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers

# ---------------------------
# Config: synthetic dataset
# ---------------------------
n_samples = 500_000     
n_features = 2048       # input features
n_classes = 10          # output classes
batch_size = 512        # large batch size to stress GPU/CPU
epochs = 10             # number of epochs increases runtime

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# Generate synthetic dataset
# ---------------------------
print("Generating synthetic dataset...")
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, n_classes, size=(n_samples,))
y = tf.keras.utils.to_categorical(y, num_classes=n_classes)

# ---------------------------
# Define a large model
# ---------------------------
print("Building model...")
model = models.Sequential()
model.add(layers.Input(shape=(n_features,)))
model.add(layers.Dense(4096, activation='relu', input_shape=(n_features,)))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# Benchmark training time
# ---------------------------
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

start_time = time.time()
history = model.fit(X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)
end_time = time.time()

# ---------------------------
# Show execution time
# ---------------------------
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = total_seconds % 60
print(f"\nTotal training time for {epochs} epochs: {minutes}' {seconds:.2f}''")