import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from attacks.badnets import BadNetsDataset, add_trigger
from models.cnn import SimpleCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005

POISON_RATE = 0.10
TARGET_CLASS = 0

TRIGGER_VALUE = 1.0

TRIGGER_SIZES = [2, 3, 4, 5]

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TRIGGER SIZE VARIATION EXPERIMENT")
print("=" * 70)

print(f"Using device: {device}")
print(f"Poisoning rate: {POISON_RATE * 100:.0f}%")
print(f"Target class: {TARGET_CLASS}")
print(f"Trigger sizes: {TRIGGER_SIZES}")
print()


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# LOAD CIFAR-10
# ============================================================

print("Loading CIFAR-10...")

train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print()


# ============================================================
# CLEAN TEST ACCURACY
# ============================================================

def evaluate_clean_accuracy(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# ============================================================
# ASR EVALUATION
# ============================================================

def evaluate_asr(model, dataset, trigger_size):

    model.eval()

    successful = 0
    total = 0

    # Only non-target-class images are used for ASR.
    # This follows the standardized evaluation used
    # for the 10% BadNets baseline.

    for index in range(len(dataset)):

        image, original_label = dataset[index]

        if original_label == TARGET_CLASS:
            continue

        # Add trigger to the clean image.
        poisoned_image = add_trigger(
            image.clone(),
            trigger_size=trigger_size,
            trigger_value=TRIGGER_VALUE
        )

        poisoned_image = poisoned_image.unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(poisoned_image)
            prediction = output.argmax(dim=1).item()

        if prediction == TARGET_CLASS:
            successful += 1

        total += 1

    if total == 0:
        return 0.0, 0, 0

    asr = 100.0 * successful / total

    return asr, successful, total


# ============================================================
# TRAINING
# ============================================================

def train_model(trigger_size):

    print("=" * 70)
    print(f"TRAINING WITH TRIGGER SIZE: {trigger_size}x{trigger_size}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create BadNets dataset
    # --------------------------------------------------------

    poisoned_train_dataset = BadNetsDataset(
        train_dataset,
        poison_rate=POISON_RATE,
        target_class=TARGET_CLASS,
        trigger_size=trigger_size,
        trigger_value=TRIGGER_VALUE,
        seed=SEED
    )

    poisoned_count = int(
        len(train_dataset) * POISON_RATE
    )

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Poisoned samples: {poisoned_count}")
    print(f"Clean samples: {len(train_dataset) - poisoned_count}")
    print()

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        poisoned_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SimpleCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.5
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_accuracy = 0.0

    model_path = os.path.join(
        MODEL_DIR,
        f"trigger_size_{trigger_size}x{trigger_size}_best.pth"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _, _ in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_accuracy = (
            100.0 * correct / total
        )

        clean_accuracy = evaluate_clean_accuracy(
            model,
            test_dataset
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Clean Acc: {clean_accuracy:.2f}%"
        )

        # Save best model according to clean accuracy
        if clean_accuracy > best_accuracy:

            best_accuracy = clean_accuracy

            torch.save(
                model.state_dict(),
                model_path
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    final_clean_accuracy = evaluate_clean_accuracy(
        model,
        test_dataset
    )

    final_asr, successful, total = evaluate_asr(
        model,
        test_dataset,
        trigger_size
    )

    print()
    print(f"Best Clean Accuracy: {final_clean_accuracy:.2f}%")
    print(
        f"ASR: {final_asr:.2f}% "
        f"({successful}/{total})"
    )
    print(f"Model saved: {model_path}")
    print()

    return {
        "trigger_size": trigger_size,
        "poisoned_samples": poisoned_count,
        "clean_accuracy": final_clean_accuracy,
        "asr": final_asr,
        "successful": successful,
        "total": total,
        "model_path": model_path
    }


# ============================================================
# RUN ALL TRIGGER SIZES
# ============================================================

results = []

for trigger_size in TRIGGER_SIZES:

    result = train_model(trigger_size)

    results.append(result)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    RESULTS_DIR,
    "trigger_size_results.txt"
)

with open(result_file, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("TRIGGER SIZE VARIATION RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"Dataset: CIFAR-10\n")
    f.write(f"Model: SimpleCNN\n")
    f.write(f"Poisoning Rate: {POISON_RATE * 100:.0f}%\n")
    f.write(f"Target Class: {TARGET_CLASS}\n")
    f.write(f"Trigger Value: {TRIGGER_VALUE}\n")
    f.write("Trigger Position: Bottom-right\n")
    f.write("Trigger Shape: Square\n\n")

    f.write(
        "Trigger Size | Poisoned Samples | "
        "Clean Accuracy (%) | ASR (%)\n"
    )

    f.write("-" * 70 + "\n")

    for result in results:

        f.write(
            f"{result['trigger_size']}x"
            f"{result['trigger_size']} | "
            f"{result['poisoned_samples']} | "
            f"{result['clean_accuracy']:.2f} | "
            f"{result['asr']:.2f}\n"
        )

print("=" * 70)
print("TRIGGER SIZE EXPERIMENT COMPLETED")
print("=" * 70)

print()
print("Final Results:")
print()

print(
    f"{'Trigger Size':<15}"
    f"{'Poisoned':<15}"
    f"{'Clean Accuracy':<20}"
    f"{'ASR':<10}"
)

print("-" * 60)

for result in results:

    print(
        f"{result['trigger_size']}x"
        f"{result['trigger_size']:<13}"
        f"{result['poisoned_samples']:<15}"
        f"{result['clean_accuracy']:<20.2f}"
        f"{result['asr']:.2f}%"
    )

print()
print(f"Results saved to:")
print(result_file)