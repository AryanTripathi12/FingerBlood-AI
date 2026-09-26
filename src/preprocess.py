import cv2
import numpy as np


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load image:", image_path)
        return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Resize
    resized = cv2.resize(
        gray,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    # 3. Normalize pixel values
    normalized = resized / 255.0

    return normalized


# Test our function
image_path = "sample_fingerprint.jpg"

processed = preprocess_image(image_path)

if processed is not None:
    print("Original image loaded successfully!")
    print("Processed shape:", processed.shape)
    print("Minimum pixel value:", processed.min())
    print("Maximum pixel value:", processed.max())