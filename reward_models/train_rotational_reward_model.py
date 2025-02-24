# Starting Training...
# Epoch [1/10], Loss: 221.6639, Accuracy: 0.7188
# Epoch [2/10], Loss: 194.1441, Accuracy: 0.7723
# Epoch [3/10], Loss: 178.2068, Accuracy: 0.7948
# Epoch [4/10], Loss: 138.3414, Accuracy: 0.8551
# Epoch [5/10], Loss: 119.9438, Accuracy: 0.8741
# Epoch [6/10], Loss: 113.4021, Accuracy: 0.8831
# Epoch [7/10], Loss: 104.8475, Accuracy: 0.8945
# Epoch [8/10], Loss: 97.6052, Accuracy: 0.9028
# Epoch [9/10], Loss: 95.9295, Accuracy: 0.9030
# Epoch [10/10], Loss: 89.3838, Accuracy: 0.9103
# Evaluating on Test Set...
# Test Accuracy: 0.9206
# Model saved as is_rotational_classifier.pth

# Starting Training...
# Epoch [1/10], Loss: 44.4117, Accuracy: 0.9595
# Epoch [2/10], Loss: 16.7685, Accuracy: 0.9920
# Epoch [3/10], Loss: 14.3825, Accuracy: 0.9930
# Epoch [4/10], Loss: 14.3388, Accuracy: 0.9934
# Epoch [5/10], Loss: 14.4424, Accuracy: 0.9932
# Epoch [6/10], Loss: 13.7049, Accuracy: 0.9934
# Epoch [7/10], Loss: 12.7536, Accuracy: 0.9931
# Epoch [8/10], Loss: 11.2195, Accuracy: 0.9933
# Epoch [9/10], Loss: 7.5279, Accuracy: 0.9955
# Epoch [10/10], Loss: 4.4099, Accuracy: 0.9970
# Evaluating on Test Set...
# Test Accuracy: 0.9956
# Model saved as is_horizontal_classifier.pth

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# =============================
# 1. Custom Dataset Loader
# =============================
class SymmetryDataset(Dataset):
    def __init__(self, root_dir="/users/zzhan513/data/zzhan513/visual_reasoning/symmetric_training_imgs/horizontal_contrast_pairs", transform=None):
        """
        root_dir: Path to dataset folder.
                  It should contain two subfolders:
                  - "preferred/" for symmetric images (label=1)
                  - "rejected/" for asymmetric images (label=0)
        transform: Torchvision transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        # Load symmetric images
        sym_dir = os.path.join(root_dir, "preferred")
        for file in os.listdir(sym_dir):
            self.data.append((os.path.join(sym_dir, file), 1))  # Label 1
        
        asym_dir = os.path.join(root_dir, "rejected")
        for file in os.listdir(asym_dir):
            self.data.append((os.path.join(asym_dir, file), 0))  # Label 0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# =============================
# 2. Data Transformations
# =============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomRotation(10),  # Slight rotation for robustness
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images randomly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Change brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# =============================
# 3. Load Dataset
# =============================
dataset_path = "/users/zzhan513/data/zzhan513/visual_reasoning/symmetric_training_imgs/horizontal_contrast_pairs"
dataset = SymmetryDataset(root_dir=dataset_path, transform=transform)

# Train-Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# =============================
# 4. Define the Symmetry Classifier
# =============================
class ResNetSymmetryClassifier(nn.Module):
    def __init__(self):
        super(ResNetSymmetryClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# =============================
# 5. Training Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetSymmetryClassifier().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# =============================
# 6. Training Loop
# =============================
num_epochs = 10

print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# =============================
# 7. Testing Loop
# =============================
print("Evaluating on Test Set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        outputs = model(images)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# =============================
# 8. Save the Model
# =============================
torch.save(model.state_dict(), "is_horizontal_classifier.pth")
print("Model saved as is_horizontal_classifier.pth")