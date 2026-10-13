import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset, add_trigger


# ==================================================
# Configuration
# ==================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005

POISON_RATE = 0.10
TARGET_CLASS = 0

TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Reproducibility
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Clean Accuracy Evaluation
# ==================================================

def evaluate_clean_accuracy(model, data_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# ==================================================
# Attack Success Rate Evaluation
# ==================================================

def evaluate_asr(model, test_dataset):

    model.eval()

    successful_attacks = 0
    total_attacked = 0

    with torch.no_grad():

        for image, original_label in test_dataset:

            # Do not attack target-class images
            if original_label == TARGET_CLASS:
                continue

            # Use the SAME trigger configuration
            triggered_image = add_trigger(
                image,
                trigger_size=TRIGGER_SIZE,
                trigger_value=TRIGGER_VALUE
            )

            triggered_image = triggered_image.unsqueeze(0)
            triggered_image = triggered_image.to(DEVICE)

            output = model(triggered_image)

            prediction = output.argmax(dim=1).item()

            if prediction == TARGET_CLASS:
                successful_attacks += 1

            total_attacked += 1

    asr = 100.0 * successful_attacks / total_attacked

    return asr, successful_attacks, total_attacked


# ==================================================
# Main
# ==================================================

def main():

    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    # ==================================================
    # Transforms
    # ==================================================

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

    # ==================================================
    # Clean training dataset
    # ==================================================

    clean_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # ==================================================
    # Poisoned training dataset
    # ==================================================

    poisoned_train_dataset = BadNetsDataset(
        dataset=clean_train_dataset,
        poison_rate=POISON_RATE,
        target_class=TARGET_CLASS,
        trigger_size=TRIGGER_SIZE,
        trigger_value=TRIGGER_VALUE,
        seed=SEED
    )

    # ==================================================
    # Clean test dataset
    # ==================================================

    clean_test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # ==================================================
    # Data loaders
    # ==================================================

    train_loader = DataLoader(
        poisoned_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ==================================================
    # Model
    # ==================================================

    model = SimpleCNN(
        num_classes=10
    ).to(DEVICE)

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

    best_clean_accuracy = 0.0
    best_epoch = 0

    print("\nStarting BadNets training...")
    print("=" * 60)

    # ==================================================
    # Training
    # ==================================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _, _ in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # Evaluate current model
        clean_accuracy = evaluate_clean_accuracy(
            model,
            clean_test_loader
        )

        scheduler.step()

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Clean Test Accuracy: {clean_accuracy:.2f}%")
        print("=" * 60)

        # ==================================================
        # Save best checkpoint
        # ==================================================

        if clean_accuracy > best_clean_accuracy:

            best_clean_accuracy = clean_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "clean_accuracy": clean_accuracy,
                    "poison_rate": POISON_RATE,
                    "target_class": TARGET_CLASS,
                    "trigger_size": TRIGGER_SIZE,
                    "trigger_value": TRIGGER_VALUE
                },
                "models/backdoored_cnn_best.pth"
            )

            print(
                "New best backdoored model saved! "
                f"CA: {clean_accuracy:.2f}%"
            )

    # ==================================================
    # Reload BEST checkpoint
    # ==================================================

    print("\nReloading best backdoored checkpoint...")

    checkpoint = torch.load(
        "models/backdoored_cnn_best.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # ==================================================
    # Evaluate BEST checkpoint
    # ==================================================

    final_clean_accuracy = evaluate_clean_accuracy(
        model,
        clean_test_loader
    )

    asr, successful_attacks, total_attacked = evaluate_asr(
        model,
        clean_test_dataset
    )

    # ==================================================
    # Final results
    # ==================================================

    print("\n" + "=" * 60)
    print("FINAL BADNETS RESULTS")
    print("=" * 60)

    print(f"Best Epoch: {best_epoch}")

    print(
        f"Clean Accuracy: "
        f"{final_clean_accuracy:.2f}%"
    )

    print(
        f"Attack Success Rate: "
        f"{asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{successful_attacks}/{total_attacked}"
    )

    print(
        f"Poison Rate: "
        f"{POISON_RATE * 100:.0f}%"
    )

    print(
        f"Target Class: "
        f"{TARGET_CLASS}"
    )

    print(
        f"Trigger Size: "
        f"{TRIGGER_SIZE}x{TRIGGER_SIZE}"
    )

    print(
        f"Trigger Value: "
        f"{TRIGGER_VALUE}"
    )

    print(
        "Model: "
        "SimpleCNN"
    )

    print(
        "Checkpoint: "
        "models/backdoored_cnn_best.pth"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()