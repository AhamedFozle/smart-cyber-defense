import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet18 import ResNet18CIFAR10


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_WORKERS = 0

MODEL_PATH = "models/resnet18_cifar10_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SETUP
# ============================================================

print("=" * 60)
print("CIFAR-10 RESNET-18 TRAINING")
print("=" * 60)

print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Training on CPU.")


# Create model directory if necessary
os.makedirs("models", exist_ok=True)


# ============================================================
# CIFAR-10 DATASET
# ============================================================

print("\nLoading CIFAR-10 dataset...")

# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating ResNet-18 model...")

model = ResNet18CIFAR10(num_classes=10)
model = model.to(DEVICE)

print("ResNet-18 created successfully!")


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# TESTING FUNCTION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    accuracy = 100.0 * correct / total

    return loss, accuracy


# ============================================================
# TRAINING
# ============================================================

print("\nStarting ResNet-18 training...")
print("=" * 60)

best_accuracy = 0.0

training_start = time.time()


for epoch in range(EPOCHS):

    print("\n" + "-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\nEpoch Results")
    print(f"Training Loss:      {train_loss:.4f}")
    print(f"Training Accuracy:  {train_accuracy:.2f}%")
    print(f"Test Loss:          {test_loss:.4f}")
    print(f"Test Accuracy:      {test_accuracy:.2f}%")

    # Save best model
    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "accuracy": best_accuracy,
                "epoch": epoch + 1
            },
            MODEL_PATH
        )

        print(
            f"\nBest ResNet-18 model saved!"
            f" Accuracy: {best_accuracy:.2f}%"
        )


# ============================================================
# TRAINING COMPLETED
# ============================================================

training_time = time.time() - training_start

print("\n" + "=" * 60)
print("RESNET-18 TRAINING COMPLETED")
print("=" * 60)

print(f"Best test accuracy: {best_accuracy:.2f}%")
print(f"Training time: {training_time:.2f} seconds")

print("\nModel saved at:")
print(MODEL_PATH)

print("=" * 60)