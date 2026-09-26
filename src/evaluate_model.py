import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model(
    "models/fingerprint_blood_group_model.keras"
)

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

test_dir = "dataset_raw/test"

X_test = []
y_test = []

# Load test images
for label, class_name in enumerate(class_names):

    class_path = os.path.join(test_dir, class_name)

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (128, 128)
        )

        image = image / 255.0

        X_test.append(image)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Add channel dimension
X_test = X_test.reshape(
    -1,
    128,
    128,
    1
)

print("Test images:", len(X_test))

# Predictions
predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# Classification report
print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)

# Confusion matrix
print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)