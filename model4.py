# model4.py
# BERT + ResNet50 + Attention-based fusion

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        alpha = torch.sigmoid(self.attn(concat))
        return alpha * x1 + (1 - alpha) * x2

class MultiModalNet4(nn.Module):
    def __init__(self):
        super(MultiModalNet4, self).__init__()
        self.text_proj = nn.Linear(384, 512)
        self.img_proj = nn.Linear(768, 512)
        self.fusion = AttentionFusion(512)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
        )

    def forward(self, text_feat, img_feat):
        text_out = self.text_proj(text_feat)
        img_out = self.img_proj(img_feat)
        fused = self.fusion(text_out, img_out)
        return self.classifier(fused)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet4().to(device)
    print(f"Model loaded on {device}")
    print(model)