# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model1 import MultiModalNet1
from model2 import MultiModalNet2
from model3 import MultiModalNet3
from model4 import MultiModalNet4
from model5 import MultiModalNet5
from model6 import MultiModalNet6
import numpy as np
from collections import Counter

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class template
class MMHSDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.text_embeds = data['text']
        self.image_embeds = data['image']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text_embeds[idx], dtype=torch.float32),
            torch.tensor(self.image_embeds[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# Class weight computation
def compute_class_weights(labels):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)
    weights = [total / (num_classes * label_counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)

# Generic training function
def train_model(model, train_loader, val_loader, model_name, num_epochs=10):
    model = model.to(device)

    # Get all training labels to compute weights
    all_train_labels = []
    for _, _, label in train_loader.dataset:
        all_train_labels.append(label.item())
    class_weights = compute_class_weights(all_train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for text, img, label in train_loader:
            text, img, label = text.to(device), img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(text, img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for text, img, label in val_loader:
                text, img, label = text.to(device), img.to(device), label.to(device)
                outputs = model(text, img)
                loss = criterion(outputs, label)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{model_name}_loss.jpg")

    # Save model
    torch.save(model.state_dict(), f"{model_name}.pth")


# Dataloaders config
def get_loaders(preprocessed_path, batch_size=64):
    train_set = MMHSDataset(f"{preprocessed_path}_train.npz")
    val_set = MMHSDataset(f"{preprocessed_path}_val.npz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Train all 6 models
for i, ModelClass in enumerate([MultiModalNet1, MultiModalNet2, MultiModalNet3, MultiModalNet4, MultiModalNet5, MultiModalNet6], 1):
    print(f"\n--- Training Model{i} ---")
    train_loader, val_loader = get_loaders(f"preprocessed{i}")
    model = ModelClass()
    train_model(model, train_loader, val_loader, f"model{i}")
