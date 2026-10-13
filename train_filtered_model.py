import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
    extract_cnn_features,
    run_isolation_forest
)

from defenses.poison_filter import (
    create_filtered_dataset
)


# ============================================
# Configuration
# ============================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0

NUM_CLEAN = 4500
NUM_POISONED = 500

CONTAMINATION = 0.10

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)


# ============================================
# Dataset transforms
# ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================
# Load CIFAR-10 training data
# ============================================

print(f"Using device: {DEVICE}")

clean_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================
# Create poisoned training dataset
# ============================================

poisoned_train_dataset = BadNetsDataset(
    dataset=clean_train_dataset,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    trigger_size=3,
    trigger_value=1.0,
    seed=SEED
)


# ============================================
# Load backdoored model for detection
# ============================================

detector_model = SimpleCNN(
    num_classes=10
).to(DEVICE)

checkpoint = torch.load(
    "models/backdoored_cnn_best.pth",
    map_location=DEVICE
)

detector_model.load_state_dict(
    checkpoint["model_state_dict"]
)

detector_model.eval()

print("Backdoored CNN loaded for poison detection.")


# ============================================
# Select samples for inspection
# ============================================

selected_indices = get_stratified_indices(
    dataset=poisoned_train_dataset,
    num_clean=NUM_CLEAN,
    num_poisoned=NUM_POISONED,
    seed=SEED
)


# ============================================
# Extract features
# ============================================

features, true_labels = extract_cnn_features(
    model=detector_model,
    dataset=poisoned_train_dataset,
    device=DEVICE,
    indices=selected_indices
)


# ============================================
# Detect suspicious samples
# ============================================

predicted_labels = run_isolation_forest(
    features=features,
    contamination=CONTAMINATION,
    random_state=SEED
)


# ============================================
# Remove suspicious samples
# ============================================

filtered_train_dataset, removed_indices = create_filtered_dataset(
    dataset=poisoned_train_dataset,
    selected_indices=selected_indices,
    predicted_labels=predicted_labels
)


# ============================================
# Create DataLoader
# ============================================

train_loader = DataLoader(
    filtered_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================
# Load clean test dataset
# ============================================

clean_test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    clean_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# Create fresh model
# ============================================

model = SimpleCNN(
    num_classes=10
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================
# Evaluation function
# ============================================

def evaluate_clean_accuracy():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ============================================
# Train filtered model
# ============================================

print("\n" + "=" * 60)
print("TRAINING FILTERED / RECOVERED MODEL")
print("=" * 60)

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        # BadNetsDataset returns:
        # image, training_label, is_poisoned, original_label
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    clean_accuracy = evaluate_clean_accuracy()

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%")
    print("-" * 50)

    if clean_accuracy > best_accuracy:

        best_accuracy = clean_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "clean_accuracy": clean_accuracy
            },
            "models/filtered_cnn_best.pth"
        )

        print(
            f"New best filtered model saved! "
            f"CA: {clean_accuracy:.2f}%"
        )


print("\n" + "=" * 60)
print("FILTERED MODEL TRAINING COMPLETED")
print("=" * 60)
print(f"Best Clean Accuracy: {best_accuracy:.2f}%")
print(
    "Model saved at: "
    "models/filtered_cnn_best.pth"
)
print("=" * 60)