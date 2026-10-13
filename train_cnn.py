import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from utils.data_loader import get_data_loaders


# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    model = SimpleCNN(num_classes=10).to(DEVICE)

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

    best_accuracy = 0.0

    print("\nStarting CNN training...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader):
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

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | "
                    f"Batch [{batch_index + 1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        scheduler.step()

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("=" * 60)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": test_accuracy
                },
                "models/simple_cnn_best.pth"
            )

            print(f"✓ New best model saved: {best_accuracy:.2f}%")

    print("\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()