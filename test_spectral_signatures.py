import torch

from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
    calculate_detection_metrics
)

from defenses.spectral_signatures import (
    extract_features,
    run_spectral_signatures,
    run_spectral_threshold_detection
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

POISON_RATE = 0.10

TARGET_CLASS = 0

NUM_CLEAN = 4500

NUM_POISONED = 500

REFERENCE_SAMPLES = 5000

THRESHOLD_PERCENTILE = 99.0

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD BACKDOORED MODEL
# ============================================================

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


print(
    "\nBackdoored CNN model loaded successfully!"
)

print(
    f"Model Clean Accuracy: "
    f"{checkpoint['clean_accuracy']:.2f}%"
)


# ============================================================
# CIFAR-10 TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# LOAD CIFAR-10
# ============================================================

clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# ============================================================
# CREATE BADNETS DATASET
# ============================================================

poisoned_dataset = BadNetsDataset(
    dataset=clean_dataset,

    poison_rate=POISON_RATE,

    target_class=TARGET_CLASS,

    trigger_size=TRIGGER_SIZE,

    trigger_value=TRIGGER_VALUE,

    seed=SEED
)


# ============================================================
# CREATE EVALUATION SUBSET
# ============================================================

selected_indices = get_stratified_indices(
    dataset=poisoned_dataset,

    num_clean=NUM_CLEAN,

    num_poisoned=NUM_POISONED,

    seed=SEED
)


# ============================================================
# CREATE CLEAN REFERENCE INDICES
#
# We use only clean samples from the BadNets dataset.
# These samples are used to establish the threshold.
# ============================================================

clean_indices = []

for index in range(
    len(poisoned_dataset)
):

    _, _, is_poisoned, _ = (
        poisoned_dataset[index]
    )

    if not is_poisoned:

        clean_indices.append(index)

    if len(clean_indices) >= REFERENCE_SAMPLES:

        break


print(
    "\nClean reference set created"
)

print(
    f"Reference samples: "
    f"{len(clean_indices)}"
)


# ============================================================
# EXTRACT CLEAN REFERENCE FEATURES
# ============================================================

reference_features, reference_labels = (
    extract_features(
        model=model,

        dataset=poisoned_dataset,

        device=DEVICE,

        indices=clean_indices
    )
)


# ============================================================
# EXTRACT EVALUATION FEATURES
# ============================================================

evaluation_features, true_labels = (
    extract_features(
        model=model,

        dataset=poisoned_dataset,

        device=DEVICE,

        indices=selected_indices
    )
)


# ============================================================
# ORIGINAL CONTROLLED EXPERIMENT
# ============================================================

print("\n" + "=" * 60)

print(
    "CONTROLLED SPECTRAL SIGNATURES"
)

print("=" * 60)


controlled_predictions, controlled_scores = (
    run_spectral_signatures(
        features=evaluation_features,

        num_poisoned=NUM_POISONED
    )
)


controlled_metrics = (
    calculate_detection_metrics(
        true_labels=true_labels,

        predicted_labels=controlled_predictions
    )
)


print(
    f"\nDetection Accuracy: "
    f"{controlled_metrics['Detection Accuracy']:.2f}%"
)

print(
    f"TPR: "
    f"{controlled_metrics['TPR']:.2f}%"
)

print(
    f"FPR: "
    f"{controlled_metrics['FPR']:.2f}%"
)


# ============================================================
# THRESHOLD-BASED EXPERIMENT
# ============================================================

print("\n" + "=" * 60)

print(
    "THRESHOLD-BASED SPECTRAL SIGNATURES"
)

print("=" * 60)


threshold_predictions, threshold_scores, threshold = (
    run_spectral_threshold_detection(

        reference_features=reference_features,

        evaluation_features=evaluation_features,

        percentile=THRESHOLD_PERCENTILE
    )
)


threshold_metrics = (
    calculate_detection_metrics(
        true_labels=true_labels,

        predicted_labels=threshold_predictions
    )
)


# ============================================================
# FINAL THRESHOLD RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "THRESHOLD-BASED RESULTS"
)

print("=" * 60)

print(
    f"Detection Accuracy: "
    f"{threshold_metrics['Detection Accuracy']:.2f}%"
)

print(
    f"TPR: "
    f"{threshold_metrics['TPR']:.2f}%"
)

print(
    f"FPR: "
    f"{threshold_metrics['FPR']:.2f}%"
)

print(
    f"TP: "
    f"{threshold_metrics['TP']}"
)

print(
    f"TN: "
    f"{threshold_metrics['TN']}"
)

print(
    f"FP: "
    f"{threshold_metrics['FP']}"
)

print(
    f"FN: "
    f"{threshold_metrics['FN']}"
)

print(
    f"Threshold: "
    f"{threshold:.6f}"
)

print(
    f"Reference Percentile: "
    f"{THRESHOLD_PERCENTILE:.2f}%"
)

print(
    f"Samples Flagged: "
    f"{threshold_predictions.sum()}"
)

print("=" * 60)


# ============================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================

print(
    "\nExperimental Configuration"
)

print("-" * 60)

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
    f"Clean Evaluation Samples: "
    f"{NUM_CLEAN}"
)

print(
    f"Poisoned Evaluation Samples: "
    f"{NUM_POISONED}"
)

print(
    f"Clean Reference Samples: "
    f"{REFERENCE_SAMPLES}"
)

print(
    f"Total Evaluation Samples: "
    f"{NUM_CLEAN + NUM_POISONED}"
)

print(
    f"Feature Dimension: "
    f"{evaluation_features.shape[1]}"
)

print("=" * 60)