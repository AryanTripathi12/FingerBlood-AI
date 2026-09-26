import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix from our evaluation
cm = np.array([
    [55, 0, 1, 0, 0, 0, 4, 1],
    [0, 95, 0, 1, 0, 4, 5, 0],
    [2, 0, 57, 0, 1, 0, 2, 0],
    [0, 9, 0, 61, 1, 3, 0, 3],
    [0, 7, 4, 2, 46, 8, 0, 1],
    [0, 1, 0, 0, 0, 72, 0, 0],
    [2, 2, 0, 0, 0, 0, 77, 4],
    [2, 0, 0, 1, 0, 0, 8, 56]
])

classes = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fingerprint Blood Group CNN - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.show()