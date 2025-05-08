# models/model.py

import torch
import torch.nn as nn

class LipReadingModel(nn.Module):
    def __init__(self, vocab_size):
        super(LipReadingModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=64*25*12,  # 64 channels, 25x12 size after 2 pools
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size*seq_len, c, h, w)
        
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(cnn_features)
        out = self.fc(lstm_out)
        
        return out
