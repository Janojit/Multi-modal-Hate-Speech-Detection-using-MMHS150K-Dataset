# evaluate.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os

from model1 import MultiModalNet1
from model2 import MultiModalNet2
from model3 import MultiModalNet3
from model4 import MultiModalNet4
from model5 import MultiModalNet5
from model6 import MultiModalNet6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_classes = [
    MultiModalNet1,
    MultiModalNet2,
    MultiModalNet3,
    MultiModalNet4,
    MultiModalNet5,
    MultiModalNet6
]

metric_results = {
    'Model': [],
    'Accuracy': [],
    'F1 Score': [],
    'Precision': [],
    'Recall': []
}

conf_matrices = []

for i in range(1, 7):
    print(f"Evaluating Model{i}...")
    model_path = f"model{i}.pth"
    test_data_path = f"preprocessed{i}_test.npz"

    # Load data
    data = np.load(test_data_path)
    text = torch.tensor(data['text'], dtype=torch.float32).to(device)
    img = torch.tensor(data['image'], dtype=torch.float32).to(device)
    labels = torch.tensor(data['labels'], dtype=torch.long).to(device)

    # Load model
    model = model_classes[i - 1]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(text, img)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

    # Calculate metrics
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    metric_results['Model'].append(f"Model{i}")
    metric_results['Accuracy'].append(acc)
    metric_results['F1 Score'].append(f1)
    metric_results['Precision'].append(prec)
    metric_results['Recall'].append(rec)
    conf_matrices.append(cm)

# Plot bar chart of metrics
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metric_results['Model']))
width = 0.2

ax.bar(x - 1.5 * width, metric_results['Accuracy'], width, label='Accuracy')
ax.bar(x - 0.5 * width, metric_results['F1 Score'], width, label='F1 Score')
ax.bar(x + 0.5 * width, metric_results['Precision'], width, label='Precision')
ax.bar(x + 1.5 * width, metric_results['Recall'], width, label='Recall')

ax.set_xlabel("Models")
ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metric_results['Model'])
ax.legend()

plt.tight_layout()
plt.savefig("evaluation_metrics_comparison.jpg")
plt.close()

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, cm in enumerate(conf_matrices):
    row = idx // 3
    col = idx % 3
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row][col])
    axes[row][col].set_title(f"Confusion Matrix: Model{idx+1}")
    axes[row][col].set_xlabel("Predicted")
    axes[row][col].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.jpg")
plt.close()

# Print the metrics for each model
print("Model Evaluation Metrics:")
for i in range(len(metric_results['Model'])):
    print(f"Model {i+1}:")
    print(f"  Accuracy: {metric_results['Accuracy'][i]:.4f}")
    print(f"  F1 Score: {metric_results['F1 Score'][i]:.4f}")
    print(f"  Precision: {metric_results['Precision'][i]:.4f}")
    print(f"  Recall: {metric_results['Recall'][i]:.4f}")
    print("-" * 40)

print("Evaluation complete. Metrics and confusion matrices saved.")
