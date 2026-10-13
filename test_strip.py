import numpy as np
import torch
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.strip import (
    calculate_strip_entropies,
    find_threshold,
    calculate_metrics
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TARGET_CLASS = 0

POISON_RATE = 0.10

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0

NUM_CLEAN = 500

NUM_POISONED = 500

NUM_REFERENCE_IMAGES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print(
    f"Using device: {DEVICE}"
)


# ============================================================
# LOAD MODEL
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

print(
    "Model ASR: 94.17% "
    "(official standardized BadNets evaluation)"
)


# ============================================================
# TRANSFORMATION
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
    train=False,
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
# COLLECT CLEAN SAMPLES
# ============================================================

clean_images = []

poisoned_images = []


for index in range(
    len(poisoned_dataset)
):

    image, _, is_poisoned, _ = (
        poisoned_dataset[index]
    )

    if is_poisoned:

        if len(poisoned_images) < NUM_POISONED:

            poisoned_images.append(
                image
            )

    else:

        if len(clean_images) < NUM_CLEAN:

            clean_images.append(
                image
            )

    if (
        len(clean_images) >= NUM_CLEAN
        and
        len(poisoned_images) >= NUM_POISONED
    ):
        break


clean_images = torch.stack(
    clean_images
)

poisoned_images = torch.stack(
    poisoned_images
)


print(
    f"\nClean samples: "
    f"{len(clean_images)}"
)

print(
    f"Poisoned samples: "
    f"{len(poisoned_images)}"
)


# ============================================================
# REFERENCE IMAGES
# ============================================================

reference_images = clean_images[
    :NUM_REFERENCE_IMAGES
]


print(
    f"Reference images: "
    f"{len(reference_images)}"
)


# ============================================================
# CLEAN STRIP ENTROPY
# ============================================================

clean_entropies = calculate_strip_entropies(
    model=model,

    images=clean_images,

    reference_images=reference_images,

    device=DEVICE,

    num_perturbations=NUM_REFERENCE_IMAGES
)


# ============================================================
# POISONED STRIP ENTROPY
# ============================================================

poisoned_entropies = calculate_strip_entropies(
    model=model,

    images=poisoned_images,

    reference_images=reference_images,

    device=DEVICE,

    num_perturbations=NUM_REFERENCE_IMAGES
)


# ============================================================
# ENTROPY SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "STRIP ENTROPY SUMMARY"
)

print("=" * 60)

print(
    f"Clean mean entropy: "
    f"{np.mean(clean_entropies):.4f}"
)

print(
    f"Clean std entropy: "
    f"{np.std(clean_entropies):.4f}"
)

print(
    f"Poisoned mean entropy: "
    f"{np.mean(poisoned_entropies):.4f}"
)

print(
    f"Poisoned std entropy: "
    f"{np.std(poisoned_entropies):.4f}"
)

print("=" * 60)


# ============================================================
# FIND THRESHOLD
# ============================================================

threshold, threshold_accuracy = find_threshold(
    clean_entropies=clean_entropies,

    poisoned_entropies=poisoned_entropies
)


print("\n" + "=" * 60)

print(
    "STRIP THRESHOLD"
)

print("=" * 60)

print(
    f"Selected entropy threshold: "
    f"{threshold:.4f}"
)

print(
    f"Threshold selection accuracy: "
    f"{threshold_accuracy:.2f}%"
)

print("=" * 60)


# ============================================================
# CALCULATE METRICS
# ============================================================

metrics = calculate_metrics(
    clean_entropies=clean_entropies,

    poisoned_entropies=poisoned_entropies,

    threshold=threshold
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "STRIP RESULTS"
)

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

print(
    f"TP: "
    f"{metrics['TP']}"
)

print(
    f"TN: "
    f"{metrics['TN']}"
)

print(
    f"FP: "
    f"{metrics['FP']}"
)

print(
    f"FN: "
    f"{metrics['FN']}"
)

print("=" * 60)


# ============================================================
# CONFIGURATION
# ============================================================

print(
    "\nExperimental Configuration"
)

print("-" * 60)

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
    f"Clean Samples: "
    f"{NUM_CLEAN}"
)

print(
    f"Poisoned Samples: "
    f"{NUM_POISONED}"
)

print(
    f"Reference Images: "
    f"{NUM_REFERENCE_IMAGES}"
)

print("=" * 60)