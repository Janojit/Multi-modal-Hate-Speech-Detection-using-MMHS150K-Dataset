# model1.py
# MLP combining BERT (768-dim) + ResNet50 (2048-dim)

import torch
import torch.nn as nn

class MultiModalNet1(nn.Module):
    def __init__(self):
        super(MultiModalNet1, self).__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.img_branch = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6 classes
        )

    def forward(self, text_feat, img_feat):
        text_out = self.text_branch(text_feat)
        img_out = self.img_branch(img_feat)
        combined = torch.cat((text_out, img_out), dim=1)
        return self.classifier(combined)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet1().to(device)
    print(f"Model loaded on {device}")
    print(model)
