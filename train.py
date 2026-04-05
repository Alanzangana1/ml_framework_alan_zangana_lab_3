import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN

# Förbereder bilderna - normaliserar pixelvärdena
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Laddar ner CIFAR-10 datan automatiskt
print("Laddar CIFAR-10 datan...")
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# Skapar modellen
model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Tränar modellen
print("Börjar träna...")
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/10 klar!")

# Testar noggrannheten
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Noggrannhet: {100 * correct / total:.1f}%")

# Sparar modellen
torch.save(model.state_dict(), "best_model.pth")
print("Modellen sparad till best_model.pth!")