import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cnn_model import create_model
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def load_data(data_dir, batch_size=16, train_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Split into train and val
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def train(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'deepfake_cnn.pth')
            print("✅ Saved best model!")

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    data_dir = "dataset"  # This should be the parent directory containing 'real' and 'fake' folders
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=device)

    train_loader, val_loader = load_data(data_dir)
    train(model, train_loader, val_loader, device)
