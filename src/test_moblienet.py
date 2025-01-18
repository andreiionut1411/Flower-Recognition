from torchvision import transforms
from fine_tune_mobilenet import FlowerDataset, evaluate
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
import torch
import torch.nn as nn


test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = FlowerDataset('test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = 102
model = mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load("mobilenetv3_flower_classifier.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()
_, acc, precision, recall, f1 = evaluate(model, test_loader, criterion, device)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")