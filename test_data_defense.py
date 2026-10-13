from torchvision import datasets, transforms

from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    extract_features,
    run_isolation_forest,
    calculate_detection_metrics
)


# -----------------------------------------------
# Configuration
# -----------------------------------------------

POISON_RATE = 0.10
TARGET_CLASS = 0

# Start with fewer samples for testing on CPU.
MAX_SAMPLES = 5000


# -----------------------------------------------
# Load CIFAR-10
# -----------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# -----------------------------------------------
# Create the same BadNets configuration
# -----------------------------------------------

poisoned_dataset = BadNetsDataset(
    dataset=clean_dataset,
    poison_rate=POISON_RATE,
    target_class=TARGET_CLASS,
    trigger_size=3,
    trigger_value=1.0,
    seed=42
)


# -----------------------------------------------
# Extract features
# -----------------------------------------------

features, true_labels = extract_features(
    poisoned_dataset,
    max_samples=MAX_SAMPLES
)


# -----------------------------------------------
# Run Isolation Forest
# -----------------------------------------------

predicted_labels = run_isolation_forest(
    features,
    contamination=POISON_RATE,
    random_state=42
)


# -----------------------------------------------
# Calculate metrics
# -----------------------------------------------

metrics = calculate_detection_metrics(
    true_labels,
    predicted_labels
)


# -----------------------------------------------
# Print results
# -----------------------------------------------

print("\n" + "=" * 50)
print("DATA-LEVEL DEFENSE RESULTS")
print("=" * 50)

for name, value in metrics.items():

    if isinstance(value, float):
        print(f"{name}: {value:.2f}%")
    else:
        print(f"{name}: {value}")

print("=" * 50)