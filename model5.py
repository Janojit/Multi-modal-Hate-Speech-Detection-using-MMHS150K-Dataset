# model5.py

import torch
import torch.nn as nn

class MultiModalNet5(nn.Module):
    def __init__(self):
        super(MultiModalNet5, self).__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, text_feat, img_feat):
        text_out = self.text_proj(text_feat)   # (B, 256)
        img_out = self.img_proj(img_feat)      # (B, 256)
        combined = torch.cat((text_out, img_out), dim=1)  # (B, 512)
        return self.classifier(combined)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet5().to(device)
    print(f"Model loaded on {device}")
    print(model)
