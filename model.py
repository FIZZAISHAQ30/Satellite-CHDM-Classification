import torch
import torch.nn as nn

# 🔹 CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


# 🔹 Diffusion-like Refiner
class DiffusionRefiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 🔹 Classifier (IMPROVED)
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1,1))  # 🔥 key improvement

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


# 🔥 FINAL MODEL
class CHDM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = CNNEncoder()
        self.refiner = DiffusionRefiner()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.refiner(x)
        x = self.classifier(x)
        return x