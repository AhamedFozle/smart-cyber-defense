import torch

from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
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

NUM_CLEAN = 4500
NUM_POISONED = 500

CONTAMINATION = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================
# Load model
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

print("Backdoored model loaded successfully!")


# ============================================
# Dataset transform
# ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================
# Load CIFAR-10
# ============================================

clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================
# Create BadNets dataset
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
# Create exact stratified subset
# ============================================

selected_indices = get_stratified_indices(
    dataset=poisoned_dataset,
    num_clean=NUM_CLEAN,
    num_poisoned=NUM_POISONED,
    seed=SEED
)


# ============================================
# Extract CNN features
# ============================================

features, true_labels = extract_cnn_features(
    model=model,
    dataset=poisoned_dataset,
    device=DEVICE,
    indices=selected_indices
)


# ============================================
# Run Isolation Forest
# ============================================

predicted_labels = run_isolation_forest(
    features=features,
    contamination=CONTAMINATION,
    random_state=SEED
)


# ============================================
# Calculate metrics
# ============================================

metrics = calculate_detection_metrics(
    true_labels=true_labels,
    predicted_labels=predicted_labels
)


# ============================================
# Print results
# ============================================

print("\n" + "=" * 60)
print("STRATIFIED CNN + ISOLATION FOREST RESULTS")
print("=" * 60)

print(f"Clean samples: {NUM_CLEAN}")
print(f"Poisoned samples: {NUM_POISONED}")
print(f"Contamination setting: {CONTAMINATION}")

print("-" * 60)

for name, value in metrics.items():

    if isinstance(value, float):
        print(f"{name}: {value:.2f}%")
    else:
        print(f"{name}: {value}")

print("=" * 60)