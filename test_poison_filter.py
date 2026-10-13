import torch

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

print("Backdoored model loaded successfully!")


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
# Load clean dataset
# ============================================

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
    trigger_size=3,
    trigger_value=1.0,
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
# Extract CNN features
# ============================================

features, true_labels = extract_cnn_features(
    model=model,
    dataset=poisoned_dataset,
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
# Filter suspicious samples
# ============================================

filtered_dataset, removed_indices = create_filtered_dataset(
    dataset=poisoned_dataset,
    selected_indices=selected_indices,
    predicted_labels=predicted_labels
)


# ============================================
# Analyze removed samples
# ============================================

true_poisoned_removed = 0
clean_removed = 0

for index in removed_indices:

    sample = poisoned_dataset[index]

    is_poisoned = sample[2]

    if is_poisoned:
        true_poisoned_removed += 1
    else:
        clean_removed += 1


total_removed = len(removed_indices)

if total_removed > 0:

    removal_precision = (
        true_poisoned_removed /
        total_removed
    ) * 100

else:

    removal_precision = 0.0


# ============================================
# Print final results
# ============================================

print("\n" + "=" * 60)
print("POISON FILTERING EVALUATION")
print("=" * 60)

print(f"Total removed: {total_removed}")

print(
    f"Actually poisoned removed: "
    f"{true_poisoned_removed}"
)

print(
    f"Clean samples incorrectly removed: "
    f"{clean_removed}"
)

print(
    f"Removal Precision: "
    f"{removal_precision:.2f}%"
)

print("=" * 60)