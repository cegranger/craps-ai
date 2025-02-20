import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import PIL #WHY ??
import os

# Set dataset directory
dataset_dir = "activité_1/opencv_dataset"

# Image size and parameters
img_size = (64, 64)
batch_size = 32

# Prepare the data with ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  
    validation_split=0.2  # Split 20% of the data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",  # One-hot encoding for multi-class classification
    subset="training"
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # Output layer for 6 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Précision: {accuracy * 100:.2f}%")
model.summary()

# Save the model
model.save('models/craps-ai/opencv_cnn.h5')