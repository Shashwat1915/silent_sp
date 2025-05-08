# models/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import LipReadingModel
from utils.dataset_loader import LipDataset
from torchvision import transforms

# Configs
vocab_size = 30  # Example: 26 letters + 4 special tokens
batch_size = 1
num_epochs = 10

# Dataset and Dataloader
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = LipDataset('data/lip_frames/english_hello', label=0, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipReadingModel(vocab_size).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for imgs, labels in dataloader:
        imgs = imgs.unsqueeze(1).to(device)  # add sequence dimension
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/saved_models/model_final.pth")
print("Training finished and model saved.")
