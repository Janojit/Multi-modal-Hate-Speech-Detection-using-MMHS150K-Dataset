# model2.py
# BiGRU for Text (GloVe 300-dim) + EfficientNetB0 for Image (1280-dim)

import torch
import torch.nn as nn

class MultiModalNet2(nn.Module):
    def __init__(self):
        super(MultiModalNet2, self).__init__()
        self.text_rnn = nn.GRU(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.img_branch = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # 512 from text + 512 from image
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, text_seq, img_feat):
        # Add sequence dimension if missing (B, 768) -> (B, 1, 768)
        if text_seq.dim() == 2:
            text_seq = text_seq.unsqueeze(1)  # (B, 1, 768)

        _, h_n = self.text_rnn(text_seq)  # h_n shape: (2, B, 256)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 512)

        img_out = self.img_branch(img_feat)  # (B, 512)
        combined = torch.cat((h_n, img_out), dim=1)  # (B, 1024)
        return self.classifier(combined)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet2().to(device)
    print(f"Model loaded on {device}")
    print(model)
