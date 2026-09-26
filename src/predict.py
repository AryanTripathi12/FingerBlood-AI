import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model(
    "models/fingerprint_blood_group_model.keras"
)

# These must match the training folder order
class_names = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

# Automatically find one test image
test_dir = "dataset_raw/test"

image_path = None

for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)

    if os.path.exists(class_path):
        files = os.listdir(class_path)

        if files:
            image_path = os.path.join(class_path, files[0])
            actual_class = class_name
            break

if image_path is None:
    print("No test image found.")
    exit()

print("Image:", image_path)
print("Actual Blood Group:", actual_class)

# Load image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize
image = cv2.resize(image, (128, 128))

# Normalize
image = image / 255.0

# Add dimensions:
# (128,128)
#     ↓
# (1,128,128,1)
image = image.reshape(1, 128, 128, 1)

# Predict
predictions = model.predict(image, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")