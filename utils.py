import torch

# =========================
# 🔹 DEVICE SETUP
# =========================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# =========================
# 🔹 ACCURACY FUNCTION
# =========================
def calculate_accuracy(model, loader, device):
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

    # 🔥 avoid division error
    if total == 0:
        return 0

    return 100 * correct / total