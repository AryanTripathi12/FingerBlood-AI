import os
import cv2
import numpy as np

DATASET_PATH = "dataset_raw/train"

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Classes:")
print(classes)

X = []
y = []

for label, class_name in enumerate(classes):

    class_path = os.path.join(DATASET_PATH, class_name)

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))

        image = image / 255.0

        X.append(image)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("\nDataset loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(classes))
