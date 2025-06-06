# preprocess1.py
# Text: BERT, Image: ResNet18 (using CUDA if available)

import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
IMG_DIR = 'img_resized'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/train_ids.txt'

# Initialize models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device).eval()

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()  # remove classifier

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        output = bert(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = resnet(image)
        return output.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(512)

def preprocess():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    with open(SPLIT_FILE, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    text_feats, img_feats, labels = [], [], []

    # Wrap the loop with tqdm for progress bar
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

    np.savez('preprocessed1_train.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
