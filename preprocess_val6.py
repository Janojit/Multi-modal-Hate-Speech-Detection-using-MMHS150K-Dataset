# preprocess6.py
# Text: TweetTokenizer + FastText, Image: DenseNet (with CUDA support)

import torch
from PIL import Image
import json
import os
import numpy as np
import fasttext.util
import nltk
from nltk.tokenize import TweetTokenizer
from torchvision import models, transforms
from tqdm import tqdm  # Progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

IMG_DIR = 'img_resized'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/val_ids.txt'

# Setup
nltk.download('punkt')
tokenizer = TweetTokenizer()
fasttext.util.download_model('en', if_exists='ignore')
ft_model = fasttext.load_model('cc.en.300.bin')  # runs on CPU

# Load DenseNet to device
densenet = models.densenet121(pretrained=True)
densenet.classifier = torch.nn.Identity()  # Remove the classifier head
densenet = densenet.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    tokens = tokenizer.tokenize(text)
    embeddings = [ft_model.get_word_vector(tok) for tok in tokens if tok.strip()]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(300)

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Move to GPU if available
        with torch.no_grad():
            features = densenet(image)
        return features.squeeze().cpu().numpy()  # Return to CPU as numpy array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(1024)  # Default zero vector for image features

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

        # Preprocess the text (FastText)
        text_feat = preprocess_text(entry['tweet_text'])

        # Preprocess the image (DenseNet)
        image_path = os.path.join(IMG_DIR, f"{tweet_id}.jpg")
        img_feat = preprocess_image(image_path)

        # Append features and label
        text_feats.append(text_feat)
        img_feats.append(img_feat)
        labels.append(label)

    # Save the preprocessed data
    np.savez('preprocessed6_val.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
