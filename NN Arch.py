import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

# Load dataset
digits = load_digits()
X = digits.images
y = digits.target

# Normalize pixel values to the range [0, 1]
X = X / 16.0

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential([
    Flatten(input_shape=(8, 8)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the trained model on the validation set
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
