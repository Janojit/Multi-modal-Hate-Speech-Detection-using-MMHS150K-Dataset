# explore.py

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
train_data = np.load("preprocessed1_train.npz")
val_data = np.load("preprocessed1_val.npz")
test_data = np.load("preprocessed1_test.npz")

train_labels = train_data["labels"]
val_labels = val_data["labels"]
test_labels = test_data["labels"]

# Count class frequencies
train_counts = Counter(train_labels)
val_counts = Counter(val_labels)
test_counts = Counter(test_labels)

# Get union of all classes
all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

print("Class Distribution:")

print("\nTrain:")
for cls in all_classes:
    print(f"Class {cls}: {train_counts.get(cls, 0)} instances")

print("\nValidation:")
for cls in all_classes:
    print(f"Class {cls}: {val_counts.get(cls, 0)} instances")

print("\nTest:")
for cls in all_classes:
    print(f"Class {cls}: {test_counts.get(cls, 0)} instances")

# Prepare bar plot data
train_vals = [train_counts.get(cls, 0) for cls in all_classes]
val_vals = [val_counts.get(cls, 0) for cls in all_classes]
test_vals = [test_counts.get(cls, 0) for cls in all_classes]

# Plot
x = np.arange(len(all_classes))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, train_vals, width, label='Train', color='skyblue')
plt.bar(x, val_vals, width, label='Validation', color='lightgreen')
plt.bar(x + width, test_vals, width, label='Test', color='salmon')

plt.xlabel("Class")
plt.ylabel("Number of Instances")
plt.title("Class Distribution by Split")
plt.xticks(ticks=x, labels=all_classes)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("split_class_distribution.jpg")
plt.close()
