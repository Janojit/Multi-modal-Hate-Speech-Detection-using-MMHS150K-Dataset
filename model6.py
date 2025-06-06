# model6.py
# BERT + CLIP (image encoder) with cross-modal interaction using gating

import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, t, i):
        x = torch.cat([t, i], dim=1)
        g = self.gate(x)
        return g * t + (1 - g) * i

class MultiModalNet6(nn.Module):
    def __init__(self):
        super(MultiModalNet6, self).__init__()
        self.text_proj = nn.Linear(300, 512)
        self.img_proj = nn.Linear(1024, 512)  # CLIP ViT-B/32 returns 512 dim
        self.fusion = GatedFusion(512)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
        )

    def forward(self, text_feat, img_feat):
        t = self.text_proj(text_feat)
        i = self.img_proj(img_feat)
        fused = self.fusion(t, i)
        return self.classifier(fused)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet6().to(device)
    print(f"Model loaded on {device}")
    print(model)