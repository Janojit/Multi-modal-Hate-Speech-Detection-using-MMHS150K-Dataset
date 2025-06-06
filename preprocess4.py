# preprocess4.py
# Text: Sentence-BERT, Image: ViT (with CUDA support)

import torch
from PIL import Image
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm  # Progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

IMG_DIR = 'img_resized'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/train_ids.txt'

# Load models to device
sbert = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))  # Automatically uses CUDA if available
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device).eval()

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    # Sentence-BERT processes the text and automatically moves the model to the appropriate device
    return sbert.encode(text, convert_to_numpy=True)

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)  # Move inputs to GPU
        with torch.no_grad():
            outputs = vit_model(**inputs)
        return outputs.pooler_output.squeeze().cpu().numpy()  # Move to CPU and return as numpy array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(768)  # Default zero vector for image features in case of error

def preprocess():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    with open(SPLIT_FILE, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    text_feats, img_feats, labels = [], [], []

    # Loop through all the IDs with a progress bar
    for tweet_id in tqdm(ids, desc="Processing tweets", unit="tweet"):
        if tweet_id not in data:
            continue
        entry = data[tweet_id]
        label = get_label_majority(entry['labels'])

        # Preprocess the text (Sentence-BERT)
        text_feat = preprocess_text(entry['tweet_text'])

        # Preprocess the image (ViT)
        image_path = os.path.join(IMG_DIR, f"{tweet_id}.jpg")
        img_feat = preprocess_image(image_path)

        # Append features and label
        text_feats.append(text_feat)
        img_feats.append(img_feat)
        labels.append(label)

    # Save the preprocessed data
    np.savez('preprocessed4_train.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
