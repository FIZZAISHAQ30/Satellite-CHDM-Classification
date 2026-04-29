import torch
from torchvision import transforms
from PIL import Image
from model import CHDM
from utils import get_device

# =========================
# DEVICE
# =========================
device = get_device()

# =========================
# MODEL LOAD
# =========================
model = CHDM(10)  # EuroSAT has 10 classes
model.load_state_dict(torch.load("chdm_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# CLASSES
# =========================
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
           'River', 'SeaLake']

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# =========================
# LOAD IMAGE
# =========================
img_path = r"C:\Users\PC\satellite_project\2750\PermanentCrop\PermanentCrop_1.jpg"   # 🔥 change this image

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Predicted Class:", classes[predicted.item()])