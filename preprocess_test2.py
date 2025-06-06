# preprocess2.py
# Text: DistilBERT, Image: EfficientNet-B0 (with CUDA support)

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm  # Progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
IMG_DIR = 'img_resized'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/test_ids.txt'

# Load models to device
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device).eval()
effnet = EfficientNet.from_pretrained('efficientnet-b0').to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    tokens = {key: val.to(device) for key, val in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        output = distilbert(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = effnet.extract_features(image)  # Get features from EfficientNet
        return output.mean([2, 3]).squeeze().cpu().numpy()  # Squeeze for dimensionality reduction
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(1280)  # Return a zero vector in case of error

def preprocess():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    with open(SPLIT_FILE, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    text_feats, img_feats, labels = [], [], []

    # Wrap the loop with tqdm for a progress bar
    for tweet_id in tqdm(ids, desc="Processing tweets", unit="tweet"):
        if tweet_id not in data:
            continue
        entry = data[tweet_id]
        label = get_label_majority(entry['labels'])
        text_feat = preprocess_text(entry['tweet_text'])
        image_path = os.path.join(IMG_DIR, f"{tweet_id}.jpg")
        img_feat = preprocess_image(image_path)

        text_feats.append(text_feat)
        img_feats.append(img_feat)
        labels.append(label)

    np.savez('preprocessed2_test.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
