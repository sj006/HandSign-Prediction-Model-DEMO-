import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = "ASL_Dataset"
asl_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

X = []
y = []

for idx, letter in enumerate(asl_labels):
    data = np.load(f"{dataset_path}/{letter}.npy")
    X.extend(data)
    y.extend([idx] * len(data))  # Assign numeric labels to letters

X = np.array(X)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = keras.Sequential([
    keras.layers.Input(shape=(42,)),  # 21 landmarks * 2 (x, y)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save trained model
model.save("asl_model.h5")
print("Model training complete. Model saved as asl_model.h5")
