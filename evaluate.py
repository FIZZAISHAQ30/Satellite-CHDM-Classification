import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CHDM
from utils import get_device, calculate_accuracy
import os

# =========================
# 🔹 TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# =========================
# 🔹 DATASET
# =========================
dataset_path = r"C:\Users\PC\satellite_project\2750"

dataset = datasets.ImageFolder(dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Classes:", dataset.classes)

# =========================
# 🔹 DEVICE
# =========================
device = get_device()
print("Using device:", device)

# =========================
# 🔹 MODEL LOAD
# =========================
model = CHDM(len(dataset.classes)).to(device)

model_path = r"C:\Users\PC\satellite_project\chdm_model.pth"

# 🔥 CHECK FILE EXISTS
if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit()

# 🔥 LOAD MODEL SAFELY
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

print("Model loaded successfully ✔")

# =========================
# 🔹 EVALUATION
# =========================
accuracy = calculate_accuracy(model, loader, device)

print("Final Accuracy:", accuracy)



