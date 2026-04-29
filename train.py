import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import CHDM
from utils import get_device, calculate_accuracy
import os

# =========================
# 🔹 TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# =========================
# 🔹 DATASET
# =========================
dataset_path = r"C:\Users\PC\satellite_project\2750"

dataset = datasets.ImageFolder(dataset_path, transform=transform)

# 🔥 SPLIT (train/test) → PROFESSIONAL
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# =========================
# 🔹 DEVICE & MODEL
# =========================
device = get_device()
print("Using device:", device)

model = CHDM(len(dataset.classes)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 🔹 TRAINING
# =========================
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 🔥 Evaluate on TEST data (not train)
    acc = calculate_accuracy(model, test_loader, device)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Test Accuracy: {acc:.2f}%")

# =========================
# 🔹 SAVE MODEL
# =========================
save_path = r"C:\Users\PC\satellite_project\chdm_model.pth"

torch.save(model.state_dict(), save_path)

print("Model saved successfully at:", save_path)

# =========================
# 🔹 VERIFY SAVE
# =========================
print("File exists:", os.path.exists(save_path))

cnn_accuracy = acc
print("CNN Accuracy:", cnn_accuracy)

chdm_accuracy = acc
print("CHDM Accuracy:", chdm_accuracy)