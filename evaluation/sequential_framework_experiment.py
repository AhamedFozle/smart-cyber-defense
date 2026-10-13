import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
    extract_cnn_features,
    run_isolation_forest,
    calculate_detection_metrics
)

from defenses.poison_filter import (
    create_filtered_dataset
)

from recovery.fine_pruning import FinePruner


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0

TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0

NUM_CLEAN = 4500
NUM_POISONED = 500

CONTAMINATION = 0.10

BATCH_SIZE = 64

RETRAIN_EPOCHS = 20
RECOVERY_EPOCHS = 10

LEARNING_RATE = 0.001

PRUNING_RATE = 0.30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

FILTERED_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sequential_filtered_model.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sequential_framework_final.pth"
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "sequential_framework_results.txt"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# DATA LOADING
# ============================================================

print("=" * 70)
print("SEQUENTIAL SMART CYBER DEFENCE EXPERIMENT")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Poison rate: {POISON_RATE * 100:.1f}%")
print(f"Target class: {TARGET_CLASS}")
print(f"Trigger size: {TRIGGER_SIZE}x{TRIGGER_SIZE}")
print(f"Pruning rate: {PRUNING_RATE * 100:.0f}%")


print("\nLoading CIFAR-10...")

clean_train_dataset = datasets.CIFAR10(
    root=os.path.join(
        BASE_DIR,
        "data"
    ),
    train=True,
    download=True,
    transform=transform
)

clean_test_dataset = datasets.CIFAR10(
    root=os.path.join(
        BASE_DIR,
        "data"
    ),
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# CREATE POISONED DATASET
# ============================================================

print("\nCreating 10% BadNets training dataset...")

poisoned_train_dataset = BadNetsDataset(
    dataset=clean_train_dataset,

    poison_rate=POISON_RATE,

    target_class=TARGET_CLASS,

    trigger_size=TRIGGER_SIZE,

    trigger_value=TRIGGER_VALUE,

    seed=SEED
)

print(
    f"Total training samples: "
    f"{len(poisoned_train_dataset)}"
)

print(
    f"Expected poisoned samples: "
    f"{int(len(poisoned_train_dataset) * POISON_RATE)}"
)


# ============================================================
# STEP 1
# DATA-LEVEL DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1: DATA-LEVEL DETECTION")
print("=" * 70)


print("\nLoading backdoored model...")

detector_model = SimpleCNN(
    num_classes=10
).to(DEVICE)


# Prefer standardized 10% checkpoint
checkpoint_candidates = [
    os.path.join(
        MODELS_DIR,
        "backdoored_cnn_10percent_best.pth"
    ),
    os.path.join(
        MODELS_DIR,
        "backdoored_cnn_best.pth"
    )
]

detector_checkpoint_path = None

for path in checkpoint_candidates:

    if os.path.exists(path):

        detector_checkpoint_path = path

        break


if detector_checkpoint_path is None:

    raise FileNotFoundError(
        "Could not find the standardized "
        "10% backdoored model checkpoint."
    )


checkpoint = torch.load(
    detector_checkpoint_path,
    map_location=DEVICE
)

detector_model.load_state_dict(
    checkpoint["model_state_dict"]
)

detector_model.eval()

print(
    f"Loaded: {detector_checkpoint_path}"
)


# ============================================================
# SELECT STRATIFIED DETECTION DATA
# ============================================================

selected_indices = get_stratified_indices(
    dataset=poisoned_train_dataset,

    num_clean=NUM_CLEAN,

    num_poisoned=NUM_POISONED,

    seed=SEED
)

print(
    f"\nSamples selected for detection: "
    f"{len(selected_indices)}"
)

print(
    f"Clean samples: {NUM_CLEAN}"
)

print(
    f"Poisoned samples: {NUM_POISONED}"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nExtracting CNN features...")

features, true_labels = extract_cnn_features(
    model=detector_model,

    dataset=poisoned_train_dataset,

    device=DEVICE,

    indices=selected_indices
)

print(
    f"Feature shape: {features.shape}"
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print("\nRunning Isolation Forest...")

predicted_labels = run_isolation_forest(
    features=features,

    contamination=CONTAMINATION,

    random_state=SEED
)


# ============================================================
# DETECTION METRICS
# ============================================================

detection_metrics = calculate_detection_metrics(
    true_labels,
    predicted_labels
)

print("\nDetection Results")
print("-" * 50)

print(
    f"Detection Accuracy: "
    f"{detection_metrics['accuracy']:.2f}%"
)

print(
    f"TPR: "
    f"{detection_metrics['tpr']:.2f}%"
)

print(
    f"FPR: "
    f"{detection_metrics['fpr']:.2f}%"
)

print(
    f"TP: {detection_metrics['tp']}"
)

print(
    f"TN: {detection_metrics['tn']}"
)

print(
    f"FP: {detection_metrics['fp']}"
)

print(
    f"FN: {detection_metrics['fn']}"
)


# ============================================================
# STEP 2
# REMOVE DETECTED SAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2: POISON FILTERING")
print("=" * 70)

filtered_train_dataset, removed_indices = (
    create_filtered_dataset(
        dataset=poisoned_train_dataset,

        selected_indices=selected_indices,

        predicted_labels=predicted_labels
    )
)


print(
    f"\nOriginal dataset: "
    f"{len(poisoned_train_dataset)}"
)

print(
    f"Removed samples: "
    f"{len(removed_indices)}"
)

print(
    f"Retained samples: "
    f"{len(filtered_train_dataset)}"
)


# ============================================================
# STEP 3
# RETRAIN FILTERED MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3: RETRAINING AFTER DETECTION")
print("=" * 70)


train_loader = DataLoader(
    filtered_train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


test_loader = DataLoader(
    clean_test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


filtered_model = SimpleCNN(
    num_classes=10
).to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    filtered_model.parameters(),

    lr=LEARNING_RATE
)


def evaluate_clean_accuracy(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return (
        100.0 *
        correct /
        total
    )


best_filtered_ca = 0.0

best_filtered_epoch = 0


for epoch in range(RETRAIN_EPOCHS):

    filtered_model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch in train_loader:

        images = batch[0].to(DEVICE)

        labels = batch[1].to(DEVICE)

        optimizer.zero_grad()

        outputs = filtered_model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_accuracy = (
        100.0 *
        correct /
        total
    )

    clean_accuracy = evaluate_clean_accuracy(
        filtered_model,
        test_loader
    )

    print(
        f"Epoch {epoch + 1:02d}/"
        f"{RETRAIN_EPOCHS} | "
        f"Loss: "
        f"{running_loss / len(train_loader):.4f} | "
        f"Train CA: "
        f"{train_accuracy:.2f}% | "
        f"Test CA: "
        f"{clean_accuracy:.2f}%"
    )

    if clean_accuracy > best_filtered_ca:

        best_filtered_ca = clean_accuracy

        best_filtered_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    filtered_model.state_dict(),

                "clean_accuracy":
                    clean_accuracy
            },

            FILTERED_MODEL_PATH
        )


print("\nFiltered model training completed.")

print(
    f"Best filtered CA: "
    f"{best_filtered_ca:.2f}%"
)

print(
    f"Best epoch: "
    f"{best_filtered_epoch}"
)

print(
    f"Saved: "
    f"{FILTERED_MODEL_PATH}"
)


# ============================================================
# LOAD BEST FILTERED MODEL
# ============================================================

filtered_checkpoint = torch.load(
    FILTERED_MODEL_PATH,
    map_location=DEVICE
)

filtered_model.load_state_dict(
    filtered_checkpoint["model_state_dict"]
)

filtered_model.eval()


# ============================================================
# STEP 4
# CREATE BACKDOOR TEST DATA
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4: PREPARING ASR EVALUATION")
print("=" * 70)


# We evaluate ASR using clean test images whose true
# class is NOT the target class.

asr_images = []
asr_labels = []


for index in range(
    len(clean_test_dataset)
):

    image, label = clean_test_dataset[index]

    if label != TARGET_CLASS:

        asr_images.append(image)

        asr_labels.append(label)


# ============================================================
# ASR FUNCTION
# ============================================================

def evaluate_asr(
    model
):

    model.eval()

    successful = 0

    total = 0

    with torch.no_grad():

        for start in range(
            0,
            len(asr_images),
            BATCH_SIZE
        ):

            batch_images = asr_images[
                start:
                start + BATCH_SIZE
            ]

            batch_labels = asr_labels[
                start:
                start + BATCH_SIZE
            ]

            images = torch.stack(
                batch_images
            ).to(DEVICE)

            labels = torch.tensor(
                batch_labels,
                device=DEVICE
            )

            poisoned_images = images.clone()

            # Apply the same BadNets trigger
            # used during training.
            poisoned_images[
                :,
                :,
                -TRIGGER_SIZE:,
                -TRIGGER_SIZE:
            ] = TRIGGER_VALUE

            outputs = model(
                poisoned_images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            successful += (
                predictions == TARGET_CLASS
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 *
        successful /
        total,
        successful,
        total
    )


# ============================================================
# FILTERED MODEL EVALUATION
# ============================================================

filtered_asr, filtered_successful, filtered_total = (
    evaluate_asr(
        filtered_model
    )
)


print(
    f"Filtered model CA: "
    f"{best_filtered_ca:.2f}%"
)

print(
    f"Filtered model ASR: "
    f"{filtered_asr:.2f}%"
)

print(
    f"Successful attacks: "
    f"{filtered_successful}/"
    f"{filtered_total}"
)


# ============================================================
# STEP 5
# FINE-PRUNING
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5: FINE-PRUNING")
print("=" * 70)


pruner = FinePruner(
    filtered_model,
    DEVICE
)


pruning_start = time.perf_counter()


masks = pruner.prune(
    train_loader,
    pruning_rate=PRUNING_RATE
)


pruning_time = (
    time.perf_counter()
    - pruning_start
)


pruned_channels, total_channels = (
    pruner.count_pruned_channels(
        masks
    )
)


actual_pruning_rate = (
    100.0 *
    pruned_channels /
    total_channels
)


print(
    f"\nActual pruning rate: "
    f"{actual_pruning_rate:.2f}%"
)


# ============================================================
# APPLY MASKS
# ============================================================

pruner.apply_masks(
    masks
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6: FINE-TUNING PRUNED MODEL")
print("=" * 70)


optimizer = torch.optim.Adam(
    filtered_model.parameters(),
    lr=LEARNING_RATE
)


best_recovered_ca = 0.0

best_recovery_epoch = 0


fine_tuning_start = time.perf_counter()


for epoch in range(
    RECOVERY_EPOCHS
):

    filtered_model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch in train_loader:

        images = batch[0].to(DEVICE)

        labels = batch[1].to(DEVICE)

        optimizer.zero_grad()

        outputs = filtered_model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        # IMPORTANT:
        # Keep pruned filters permanently disabled.
        pruner.reapply_masks(
            masks
        )

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    clean_accuracy = (
        evaluate_clean_accuracy(
            filtered_model,
            test_loader
        )
    )

    print(
        f"Epoch {epoch + 1:02d}/"
        f"{RECOVERY_EPOCHS} | "
        f"Loss: "
        f"{running_loss / len(train_loader):.4f} | "
        f"Clean CA: "
        f"{clean_accuracy:.2f}%"
    )

    if clean_accuracy > best_recovered_ca:

        best_recovered_ca = clean_accuracy

        best_recovery_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    filtered_model.state_dict(),

                "clean_accuracy":
                    clean_accuracy,

                "pruning_rate":
                    actual_pruning_rate
            },

            FINAL_MODEL_PATH
        )


fine_tuning_time = (
    time.perf_counter()
    - fine_tuning_start
)


# ============================================================
# LOAD FINAL BEST MODEL
# ============================================================

final_checkpoint = torch.load(
    FINAL_MODEL_PATH,
    map_location=DEVICE
)

filtered_model.load_state_dict(
    final_checkpoint["model_state_dict"]
)

filtered_model.eval()


# ============================================================
# FINAL EVALUATION
# ============================================================

final_ca = evaluate_clean_accuracy(
    filtered_model,
    test_loader
)

final_asr, final_successful, final_total = (
    evaluate_asr(
        filtered_model
    )
)


# ============================================================
# CALCULATE RECOVERY METRICS
# ============================================================

baseline_ca = 78.40
baseline_asr = 94.17


asr_reduction = (
    baseline_asr -
    final_asr
)


recovery_rate = (
    100.0 *
    asr_reduction /
    baseline_asr
)


ca_change = (
    final_ca -
    baseline_ca
)


total_recovery_time = (
    pruning_time +
    fine_tuning_time
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SEQUENTIAL FRAMEWORK RESULTS")
print("=" * 70)

print(
    f"Baseline CA: "
    f"{baseline_ca:.2f}%"
)

print(
    f"Baseline ASR: "
    f"{baseline_asr:.2f}%"
)

print("-" * 70)

print(
    f"Detection Accuracy: "
    f"{detection_metrics['accuracy']:.2f}%"
)

print(
    f"Detection TPR: "
    f"{detection_metrics['tpr']:.2f}%"
)

print(
    f"Detection FPR: "
    f"{detection_metrics['fpr']:.2f}%"
)

print(
    f"Samples removed: "
    f"{len(removed_indices)}"
)

print(
    f"Samples retained: "
    f"{len(filtered_train_dataset)}"
)

print("-" * 70)

print(
    f"Filtered model CA: "
    f"{best_filtered_ca:.2f}%"
)

print(
    f"Filtered model ASR: "
    f"{filtered_asr:.2f}%"
)

print("-" * 70)

print(
    f"Actual pruning: "
    f"{actual_pruning_rate:.2f}%"
)

print(
    f"Final recovered CA: "
    f"{final_ca:.2f}%"
)

print(
    f"Final recovered ASR: "
    f"{final_asr:.2f}%"
)

print(
    f"ASR reduction: "
    f"{asr_reduction:.2f} percentage points"
)

print(
    f"CA change: "
    f"{ca_change:+.2f} percentage points"
)

print(
    f"Recovery rate: "
    f"{recovery_rate:.2f}%"
)

print(
    f"Pruning time: "
    f"{pruning_time:.2f} seconds"
)

print(
    f"Fine-tuning time: "
    f"{fine_tuning_time:.2f} seconds"
)

print(
    f"Total recovery time: "
    f"{total_recovery_time:.2f} seconds"
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SEQUENTIAL SMART CYBER DEFENCE "
        "FRAMEWORK EXPERIMENT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Experimental configuration\n"
    )

    file.write(
        f"Poisoning rate: "
        f"{POISON_RATE * 100:.1f}%\n"
    )

    file.write(
        f"Target class: "
        f"{TARGET_CLASS}\n"
    )

    file.write(
        f"Trigger size: "
        f"{TRIGGER_SIZE}x{TRIGGER_SIZE}\n"
    )

    file.write(
        f"Detection method: "
        f"Isolation Forest\n"
    )

    file.write(
        f"Pruning rate: "
        f"{actual_pruning_rate:.2f}%\n"
    )

    file.write(
        f"Device: "
        f"{DEVICE}\n\n"
    )

    file.write(
        "Detection results\n"
    )

    file.write("-" * 50 + "\n")

    file.write(
        f"Detection Accuracy: "
        f"{detection_metrics['accuracy']:.2f}%\n"
    )

    file.write(
        f"TPR: "
        f"{detection_metrics['tpr']:.2f}%\n"
    )

    file.write(
        f"FPR: "
        f"{detection_metrics['fpr']:.2f}%\n"
    )

    file.write(
        f"TP: "
        f"{detection_metrics['tp']}\n"
    )

    file.write(
        f"TN: "
        f"{detection_metrics['tn']}\n"
    )

    file.write(
        f"FP: "
        f"{detection_metrics['fp']}\n"
    )

    file.write(
        f"FN: "
        f"{detection_metrics['fn']}\n"
    )

    file.write(
        f"Samples removed: "
        f"{len(removed_indices)}\n"
    )

    file.write(
        f"Samples retained: "
        f"{len(filtered_train_dataset)}\n\n"
    )

    file.write(
        "Sequential recovery results\n"
    )

    file.write("-" * 50 + "\n")

    file.write(
        f"Baseline CA: "
        f"{baseline_ca:.2f}%\n"
    )

    file.write(
        f"Baseline ASR: "
        f"{baseline_asr:.2f}%\n"
    )

    file.write(
        f"Filtered model CA: "
        f"{best_filtered_ca:.2f}%\n"
    )

    file.write(
        f"Filtered model ASR: "
        f"{filtered_asr:.2f}%\n"
    )

    file.write(
        f"Actual pruning rate: "
        f"{actual_pruning_rate:.2f}%\n"
    )

    file.write(
        f"Final CA: "
        f"{final_ca:.2f}%\n"
    )

    file.write(
        f"Final ASR: "
        f"{final_asr:.2f}%\n"
    )

    file.write(
        f"ASR reduction: "
        f"{asr_reduction:.2f} percentage points\n"
    )

    file.write(
        f"CA change: "
        f"{ca_change:+.2f} percentage points\n"
    )

    file.write(
        f"Recovery rate: "
        f"{recovery_rate:.2f}%\n"
    )

    file.write(
        f"Pruning time: "
        f"{pruning_time:.2f} seconds\n"
    )

    file.write(
        f"Fine-tuning time: "
        f"{fine_tuning_time:.2f} seconds\n"
    )

    file.write(
        f"Total recovery time: "
        f"{total_recovery_time:.2f} seconds\n"
    )


print(
    f"\nResults saved to:\n"
    f"{RESULT_FILE}"
)

print(
    f"Final model saved to:\n"
    f"{FINAL_MODEL_PATH}"
)