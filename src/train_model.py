import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
train_dir = "dataset_raw/train"
valid_dir = "dataset_raw/valid"
test_dir = "dataset_raw/test"

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int"
)

# Load validation data
valid_data = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int"
)

# Load test data
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int"
)

# Get class names
class_names = train_data.class_names

print("Classes:", class_names)

# Normalize images
normalization_layer = layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

valid_data = valid_data.map(
    lambda x, y: (normalization_layer(x), y)
)

test_data = test_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# Build CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(8, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model architecture
model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)

# Save model
os.makedirs("models", exist_ok=True)

model.save("models/fingerprint_blood_group_model.keras")

print("Model saved successfully!")