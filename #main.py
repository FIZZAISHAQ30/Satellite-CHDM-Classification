import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 🔥 IMPORT YOUR MODEL
from model import CHDM

# =========================
# 🔹 DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 🔹 TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# =========================
# 🔹 DATASET
# =========================
dataset_path = r"C:\Users\PC\satellite_project\2750"

dataset = datasets.ImageFolder(dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# =========================
# 🔹 MODEL
# =========================
model = CHDM(len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 🔹 TRAINING
# =========================
epochs = 66

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# =========================
# 🔹 EVALUATION (ACCURACY)
# =========================
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Accuracy:", accuracy)

# =========================
# 🔹 CONFUSION MATRIX
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 🔹 SAVE MODEL
# =========================
save_path = r"C:\Users\PC\satellite_project\chdm_model.pth"
torch.save(model.state_dict(), save_path)

print("Model saved successfully at:", save_path)