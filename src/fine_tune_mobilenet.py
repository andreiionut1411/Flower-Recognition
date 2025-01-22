import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FlowerDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label = int(image_file.split("_")[1]) - 1
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in tqdm(loader, desc="Training Progress", total=len(loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Progress", total=len(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to NumPy arrays for metrics computation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute metrics
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return running_loss / len(loader.dataset), accuracy, precision, recall, f1


def main():
	train_transforms = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	val_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	train_dataset = FlowerDataset('train', transform=train_transforms)
	val_dataset = FlowerDataset('dev', transform=val_transforms)

	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

	model = mobilenet_v3_large(pretrained=True)
	num_classes = 102
	model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	num_epochs = 20
	for epoch in range(num_epochs):
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
		print(f"Epoch {epoch+1}/{num_epochs}")
		print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
		print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

		# Adjust learning rate based on validation loss
		scheduler.step(val_loss)

	torch.save(model.state_dict(), "mobilenetv3_flower_classifier.pth")

if __name__ == "__main__":
    main()