import os
import cv2

DATASET_PATH = "dataset_raw/train"

# Find the first class folder
classes = sorted(os.listdir(DATASET_PATH))

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)

        if images:
            image_path = os.path.join(class_path, images[0])

            image = cv2.imread(image_path)

            if image is None:
                print("Could not load image")
                break

            print("Class:", class_name)
            print("Image:", image_path)
            print("Image shape:", image.shape)

            # Save a copy for us to view
            output_path = "sample_fingerprint.jpg"
            cv2.imwrite(output_path, image)

            print("Saved:", output_path)

            break