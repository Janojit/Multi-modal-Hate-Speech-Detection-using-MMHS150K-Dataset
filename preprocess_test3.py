# preprocess3.py
# Text: OCR + GloVe Avg, Image: MobileNetV2 (with CUDA support)

import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm  # Progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

IMG_DIR = 'img_resized'
OCR_DIR = 'img_txt'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/test_ids.txt'
GLOVE_FILE = 'glove.6B.300d.txt'

# Load GloVe embeddings
def load_glove(path):
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

glove = load_glove(GLOVE_FILE)

# Load MobileNetV2 model and send to GPU if available
mobilenet = models.mobilenet_v2(pretrained=True).features.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    vecs = [glove[token] for token in tokens if token in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Move image to GPU
        with torch.no_grad():
            output = mobilenet(image)  # Feature extraction
        return output.mean([2, 3]).squeeze().cpu().numpy()  # Return to CPU and flatten
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(1280)  # Default zero vector for image features in case of error

def preprocess():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    with open(SPLIT_FILE, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    text_feats, img_feats, labels = [], [], []

    # Loop through all the IDs with progress bar
    for tweet_id in tqdm(ids, desc="Processing tweets", unit="tweet"):
        if tweet_id not in data:
            continue
        entry = data[tweet_id]
        label = get_label_majority(entry['labels'])

        # Combine tweet text and OCR text
        ocr_path = os.path.join(OCR_DIR, f"{tweet_id}.txt")
        ocr_text = open(ocr_path, 'r').read() if os.path.exists(ocr_path) else ''
        full_text = entry['tweet_text'] + ' ' + ocr_text
        text_feat = preprocess_text(full_text)

        # Image processing
        image_path = os.path.join(IMG_DIR, f"{tweet_id}.jpg")
        img_feat = preprocess_image(image_path)

        # Append the processed features
        text_feats.append(text_feat)
        img_feats.append(img_feat)
        labels.append(label)

    np.savez('preprocessed3_test.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
