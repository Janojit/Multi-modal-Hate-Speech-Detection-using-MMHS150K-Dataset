# model3.py
# Transformer (DistilBERT 768-dim) + ViT (768-dim) with residual fusion

import torch
import torch.nn as nn

class MultiModalNet3(nn.Module):
    def __init__(self):
        super(MultiModalNet3, self).__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(300, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, 6)

    def forward(self, text_feat, img_feat):
        text_out = self.text_proj(text_feat)
        img_out = self.img_proj(img_feat)
        fused = text_out + img_out + self.fusion(text_out * img_out)  # residual + multiplicative fusion
        return self.classifier(fused)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet3().to(device)
    print(f"Model loaded on {device}")
    print(model)