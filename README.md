# Multimodal Hate Speech Detection (MMHS150K)

This project implements multimodal hate speech detection using both **text** (via BERT) and **image** (via ResNet18) features from the MMHS150K dataset. It includes preprocessing, training, evaluation, and visualization scripts for **six different model architectures**.

---

## 📁 Project Structure

```
multimodal-hate-speech/
├── data/                      # JSON data and ID splits (user-provided)
│   ├── MMHS150K_GT.json
│   └── splits/
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
│
├── img_resized/              # Resized tweet images (*.jpg)
│
├── preprocess{i}.py               # Preprocess train split for model i
├── preprocess_val{i}.py           # Preprocess val split for model i
├── preprocess_test{i}.py          # Preprocess test split for model i
│
├── model{i}.py                    # MultiModalNet{i} architecture (i=1 to 6)
│
├── train.py                  # Train all 6 models
├── evaluate.py               # Evaluate all 6 models on test set
├── explore.py                # Show label distribution across splits
├── requirements.txt          # Dependency list
└── README.md                 # Project documentation
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

### Folders

Ensure the following are present:

- `data/MMHS150K_GT.json`
- `data/splits/train_ids.txt`, `val_ids.txt`, `test_ids.txt`
- `img_resized/` with `{tweet_id}.jpg` files

---

## 🧪 Instructions

### 🔹 Step 1: Preprocess Features

Run the preprocessing for each model version (i=1 to 6):

```bash
python preprocess1.py && python preprocess_val1.py && python preprocess_test1.py
python preprocess2.py && python preprocess_val2.py && python preprocess_test2.py
...
python preprocess6.py && python preprocess_val6.py && python preprocess_test6.py
```

This generates:
- `preprocessed{i}_train.npz`, `preprocessed{i}_val.npz`, `preprocessed{i}_test.npz`

---

### 🔹 Step 2: Explore Data

```bash
python explore.py
```

Generates `split_class_distribution.jpg`

---

### 🔹 Step 3: Train All Models

```bash
python train.py
```

- Trains all `MultiModalNet{i}` models (i=1 to 6)
- Saves each as `model{i}.pth`
- Plots training vs validation loss as `model{i}_loss.jpg`

---

### 🔹 Step 4: Evaluate All Models

```bash
python evaluate.py
```

- Prints accuracy, precision, recall, and F1-score for each model
- Saves:
  - `evaluation_metrics_comparison.jpg`
  - `confusion_matrices_comparison.jpg`

---

## 👥 Authors

- Janojit Chakraborty — [GitHub](https://github.com/Janojit)
