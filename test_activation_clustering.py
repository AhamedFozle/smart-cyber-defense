import torch

from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
    calculate_detection_metrics
)

from defenses.activation_clustering import (
    extract_activations,
    run_activation_clustering
)


# ============================================
# Configuration
# ============================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0

NUM_CLEAN = 4500
NUM_POISONED = 500

TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

print("\nBackdoored CNN model loaded successfully!")

print(
    f"Model Clean Accuracy: "
    f"{checkpoint['clean_accuracy']:.2f}%"
)


# ============================================
# Load CIFAR-10
# ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================
# Create poisoned dataset
# ============================================

poisoned_dataset = BadNetsDataset(
    dataset=clean_dataset,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    trigger_size=TRIGGER_SIZE,
    trigger_value=TRIGGER_VALUE,
    seed=SEED
)


# ============================================
# Create stratified subset
# ============================================

selected_indices = get_stratified_indices(
    dataset=poisoned_dataset,
    num_clean=NUM_CLEAN,
    num_poisoned=NUM_POISONED,
    seed=SEED
)


# ============================================
# Extract CNN activations
# ============================================

activations, true_labels = extract_activations(
    model=model,
    dataset=poisoned_dataset,
    device=DEVICE,
    indices=selected_indices
)


# ============================================
# Activation Clustering
# ============================================

predicted_labels, cluster_info = (
    run_activation_clustering(
        activations=activations,
        true_labels=true_labels,
        random_state=SEED
    )
)


# ============================================
# Calculate detection metrics
# ============================================

metrics = calculate_detection_metrics(
    true_labels=true_labels,
    predicted_labels=predicted_labels
)


# ============================================
# Final results
# ============================================

print("\n" + "=" * 60)
print("ACTIVATION CLUSTERING RESULTS")
print("=" * 60)

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

print("=" * 60)

print("\nExperimental Configuration")
print("-" * 60)

print(f"Poison Rate: {POISON_RATE * 100:.0f}%")
print(f"Target Class: {TARGET_CLASS}")
print(f"Trigger Size: {TRIGGER_SIZE}x{TRIGGER_SIZE}")
print(f"Trigger Value: {TRIGGER_VALUE}")
print(f"Clean Samples: {NUM_CLEAN}")
print(f"Poisoned Samples: {NUM_POISONED}")
print(f"Total Samples: {NUM_CLEAN + NUM_POISONED}")
print(f"Feature Dimension: {activations.shape[1]}")

print("=" * 60)