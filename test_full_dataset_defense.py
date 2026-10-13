import time
import torch

from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    extract_cnn_features,
    run_isolation_forest,
    calculate_detection_metrics
)


# ============================================
# Configuration
# ============================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0

CONTAMINATION = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)


# ============================================
# Start timing
# ============================================

start_time = time.time()


# ============================================
# Load backdoored model
# ============================================

print(f"Using device: {DEVICE}")

model = SimpleCNN(
    num_classes=10
).to(DEVICE)

checkpoint = torch.load(
    "models/backdoored_cnn_best.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Backdoored CNN model loaded successfully!")

if "clean_accuracy" in checkpoint:
    print(
        f"Model Clean Accuracy: "
        f"{checkpoint['clean_accuracy']:.2f}%"
    )


# ============================================
# CIFAR-10 transform
# ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================
# Load training dataset
# ============================================

clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================
# Create full poisoned dataset
# ============================================

poisoned_dataset = BadNetsDataset(
    dataset=clean_dataset,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    trigger_size=3,
    trigger_value=1.0,
    seed=SEED
)


# ============================================
# Use ALL dataset indices
# ============================================

all_indices = list(
    range(len(poisoned_dataset))
)

print("\n" + "=" * 60)
print("FULL DATASET DEFENSE EXPERIMENT")
print("=" * 60)

print(f"Total samples to inspect: {len(all_indices)}")
print("Expected clean samples: 45000")
print("Expected poisoned samples: 5000")
print(f"Isolation Forest contamination: {CONTAMINATION}")
print("=" * 60)


# ============================================
# Extract CNN features from ALL samples
# ============================================

features, true_labels = extract_cnn_features(
    model=model,
    dataset=poisoned_dataset,
    device=DEVICE,
    indices=all_indices
)


# ============================================
# Run Isolation Forest
# ============================================

print("\nRunning Isolation Forest on full dataset...")

predicted_labels = run_isolation_forest(
    features=features,
    contamination=CONTAMINATION,
    random_state=SEED
)


# ============================================
# Calculate detection metrics
# ============================================

metrics = calculate_detection_metrics(
    true_labels=true_labels,
    predicted_labels=predicted_labels
)


# ============================================
# Calculate computational overhead
# ============================================

total_time = time.time() - start_time


# ============================================
# Print final results
# ============================================

print("\n" + "=" * 65)
print("FULL DATASET CNN + ISOLATION FOREST RESULTS")
print("=" * 65)

print(f"Samples inspected: {len(all_indices)}")
print(f"Feature shape: {features.shape}")
print(f"Contamination: {CONTAMINATION}")

print("-" * 65)

print(
    f"Detection Accuracy: "
    f"{metrics['Detection Accuracy']:.2f}%"
)

print(
    f"TPR: "
    f"{metrics['TPR']:.2f}%"
)

print(
    f"FPR: "
    f"{metrics['FPR']:.2f}%"
)

print(f"TP: {metrics['TP']}")
print(f"TN: {metrics['TN']}")
print(f"FP: {metrics['FP']}")
print(f"FN: {metrics['FN']}")

print("-" * 65)

print(
    f"Total Computational Time: "
    f"{total_time:.2f} seconds"
)

print(
    f"Average Time Per Sample: "
    f"{total_time / len(all_indices):.6f} seconds"
)

print("=" * 65)