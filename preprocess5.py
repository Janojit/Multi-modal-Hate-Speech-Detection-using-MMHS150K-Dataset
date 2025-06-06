# preprocess5.py
# Text: RoBERTa, Image: CLIP-ViT (OpenAI, with CUDA support)

import torch
from PIL import Image
import json
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, CLIPProcessor, CLIPModel
from tqdm import tqdm  # Progress bar

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

IMG_DIR = 'img_resized'
DATA_FILE = 'MMHS150K_GT.json'
SPLIT_FILE = 'splits/train_ids.txt'

# Load models to device
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base').to(device).eval()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_label_majority(label_list):
    return max(set(label_list), key=label_list.count)

def preprocess_text(text):
    inputs = roberta_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.pooler_output.squeeze().cpu().numpy()  # Return as numpy array to CPU

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt").to(device)  # Move to device
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        return outputs.squeeze().cpu().numpy()  # Return as numpy array to CPU
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return np.zeros(512)  # Default zero vector for image features in case of error

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

        # Preprocess the text (RoBERTa)
        text_feat = preprocess_text(entry['tweet_text'])

        # Preprocess the image (CLIP-ViT)
        image_path = os.path.join(IMG_DIR, f"{tweet_id}.jpg")
        img_feat = preprocess_image(image_path)

        # Append features and label
        text_feats.append(text_feat)
        img_feats.append(img_feat)
        labels.append(label)

    # Save the preprocessed data
    np.savez('preprocessed5_train.npz', text=np.array(text_feats), image=np.array(img_feats), labels=np.array(labels))

if __name__ == '__main__':
    preprocess()
