import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    extract_cnn_features,
    run_isolation_forest
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0
TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0

CONTAMINATION = 0.10

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)


# ============================================================
# DATA TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# LOAD BACKDOORED MODEL FOR FEATURE EXTRACTION
# ============================================================

print(f"Using device: {DEVICE}")

feature_model = SimpleCNN(
    num_classes=10
).to(DEVICE)

checkpoint = torch.load(
    "models/backdoored_cnn_best.pth",
    map_location=DEVICE
)

feature_model.load_state_dict(
    checkpoint["model_state_dict"]
)

feature_model.eval()

print("Backdoored CNN loaded for poison detection.")


# ============================================================
# LOAD CIFAR-10 TRAINING DATA
# ============================================================

clean_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================================
# CREATE POISONED DATASET
# ============================================================

poisoned_dataset = BadNetsDataset(
    dataset=clean_train_dataset,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    trigger_size=TRIGGER_SIZE,
    trigger_value=TRIGGER_VALUE,
    seed=SEED
)


# ============================================================
# FULL DATASET POISON DETECTION
# ============================================================

all_indices = list(
    range(len(poisoned_dataset))
)

print("\n" + "=" * 60)
print("FULL DATASET POISON DETECTION")
print("=" * 60)

print(
    f"Inspecting {len(all_indices)} samples..."
)


detection_start = time.time()


# Extract CNN features
features, _ = extract_cnn_features(
    model=feature_model,
    dataset=poisoned_dataset,
    device=DEVICE,
    indices=all_indices
)


# Run Isolation Forest
predicted_labels = run_isolation_forest(
    features=features,
    contamination=CONTAMINATION,
    random_state=SEED
)


detection_time = (
    time.time() - detection_start
)


# ============================================================
# CREATE FILTERED DATASET
# ============================================================

# Based on your previous experiment:
# prediction == 1 means suspicious sample
# prediction == 0 means retained sample

suspicious_indices = []

retained_indices = []


for index, prediction in zip(
    all_indices,
    predicted_labels
):

    if prediction == 1:

        suspicious_indices.append(index)

    else:

        retained_indices.append(index)


print("\n" + "=" * 60)
print("FULL DATASET FILTERING SUMMARY")
print("=" * 60)

print(
    f"Original dataset size: "
    f"{len(poisoned_dataset)}"
)

print(
    f"Suspicious samples removed: "
    f"{len(suspicious_indices)}"
)

print(
    f"Samples retained: "
    f"{len(retained_indices)}"
)

print(
    f"Detection time: "
    f"{detection_time:.2f} seconds"
)

print("=" * 60)


# ============================================================
# CREATE FILTERED TRAINING DATASET
# ============================================================

filtered_dataset = Subset(
    poisoned_dataset,
    retained_indices
)


train_loader = DataLoader(
    filtered_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# LOAD CLEAN CIFAR-10 TEST DATA
# ============================================================

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


# ============================================================
# CREATE NEW CNN MODEL
# ============================================================

model = SimpleCNN(
    num_classes=10
).to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FUNCTION: EVALUATE CLEAN ACCURACY
# ============================================================

def evaluate_clean_accuracy():

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        # Normal CIFAR-10 returns:
        # images, labels

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


    accuracy = (
        100.0 * correct / total
    )


    return accuracy


# ============================================================
# TRAIN FILTERED MODEL
# ============================================================

best_accuracy = 0.0


print("\n" + "=" * 60)
print("TRAINING FULL-DATASET FILTERED MODEL")
print("=" * 60)


for epoch in range(EPOCHS):


    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    # IMPORTANT:
    # BadNetsDataset may return multiple values.
    # We use batch[0] = images
    # We use batch[1] = training labels.
    # All additional information is ignored.

    for batch in train_loader:


        images = batch[0]

        labels = batch[1]


        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item() * labels.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    # ========================================================
    # CALCULATE TRAINING RESULTS
    # ========================================================

    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        100.0 * correct / total
    )


    # ========================================================
    # EVALUATE ON CLEAN TEST DATA
    # ========================================================

    clean_accuracy = (
        evaluate_clean_accuracy()
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Clean Test Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    print("-" * 60)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if clean_accuracy > best_accuracy:


        best_accuracy = clean_accuracy


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "clean_accuracy":
                    clean_accuracy,

                "poison_rate":
                    POISON_RATE,

                "target_class":
                    TARGET_CLASS,

                "contamination":
                    CONTAMINATION,

                "removed_samples":
                    len(suspicious_indices),

                "retained_samples":
                    len(retained_indices)
            },

            "models/full_filtered_cnn_best.pth"
        )


        print(
            "New best full-filtered model saved! "
            f"CA: {best_accuracy:.2f}%"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)

print(
    "FULL FILTERED MODEL TRAINING COMPLETED"
)

print("=" * 65)


print(
    f"Best Clean Accuracy: "
    f"{best_accuracy:.2f}%"
)


print(
    f"Suspicious samples removed: "
    f"{len(suspicious_indices)}"
)


print(
    f"Samples retained: "
    f"{len(retained_indices)}"
)


print(
    "Model saved at: "
    "models/full_filtered_cnn_best.pth"
)


print("=" * 65)