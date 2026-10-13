"""
Trigger Variation Experiment
============================

Smart Cyber Defence Framework Thesis

Purpose:
    Evaluate the sensitivity of the BadNets attack to:
        1. Trigger size
        2. Trigger position

CPU-friendly design:
    - CIFAR-10
    - SimpleCNN
    - 10% poisoning
    - 8 training epochs per configuration
    - Batch size 128
    - No expensive defense processing
    - Results saved to results/

Configurations:
    T1: 2x2 bottom-right
    T2: 3x3 bottom-right
    T3: 4x4 bottom-right
    T4: 3x3 top-left

Important:
    These are sensitivity experiments, not replacements for the
    standardized 20-epoch BadNets baseline.
"""

import os
import sys
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODEL
# ============================================================

from models.cnn import SimpleCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

POISON_RATE = 0.10
TARGET_CLASS = 0

BATCH_SIZE = 128

# CPU-friendly number of epochs.
# Keep this at 8 for the first run.
EPOCHS = 8

LEARNING_RATE = 0.001

NUM_WORKERS = 0

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TRIGGER VARIATION EXPERIMENT")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Poison rate: {POISON_RATE * 100:.1f}%")
print(f"Target class: {TARGET_CLASS}")
print(f"Epochs per configuration: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 70)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# CIFAR-10 NORMALIZATION
# ============================================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# TRIGGER FUNCTION
# ============================================================

def add_trigger(
    image,
    trigger_size=3,
    trigger_value=1.0,
    position="bottom_right"
):
    """
    Add a square trigger to a CIFAR-10 image.

    Supports:
        [C, H, W]
        [B, C, H, W]

    Positions:
        bottom_right
        top_left
        top_right
        bottom_left
    """

    image = image.clone()

    if image.dim() == 3:

        _, height, width = image.shape

        if position == "bottom_right":
            y_start = height - trigger_size
            x_start = width - trigger_size

        elif position == "top_left":
            y_start = 0
            x_start = 0

        elif position == "top_right":
            y_start = 0
            x_start = width - trigger_size

        elif position == "bottom_left":
            y_start = height - trigger_size
            x_start = 0

        else:
            raise ValueError(f"Unknown trigger position: {position}")

        image[
            :,
            y_start:y_start + trigger_size,
            x_start:x_start + trigger_size
        ] = trigger_value

    elif image.dim() == 4:

        _, _, height, width = image.shape

        if position == "bottom_right":
            y_start = height - trigger_size
            x_start = width - trigger_size

        elif position == "top_left":
            y_start = 0
            x_start = 0

        elif position == "top_right":
            y_start = 0
            x_start = width - trigger_size

        elif position == "bottom_left":
            y_start = height - trigger_size
            x_start = 0

        else:
            raise ValueError(f"Unknown trigger position: {position}")

        image[
            :,
            :,
            y_start:y_start + trigger_size,
            x_start:x_start + trigger_size
        ] = trigger_value

    else:
        raise ValueError(
            "Image must have shape [C,H,W] or [B,C,H,W]"
        )

    return image


# ============================================================
# BADNETS DATASET
# ============================================================

class TriggerVariationDataset(Dataset):
    """
    CIFAR-10 training dataset with BadNets poisoning.

    Returns:
        image
        label
        poisoned_flag
        original_label
    """

    def __init__(
        self,
        dataset,
        poison_rate=0.10,
        target_class=0,
        trigger_size=3,
        trigger_value=1.0,
        position="bottom_right",
        seed=42
    ):

        self.dataset = dataset
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.position = position

        total_samples = len(dataset)

        poison_count = int(total_samples * poison_rate)

        generator = torch.Generator()
        generator.manual_seed(seed)

        permutation = torch.randperm(
            total_samples,
            generator=generator
        )

        self.poison_indices = set(
            permutation[:poison_count].tolist()
        )

        print(
            f"Poisoned samples: "
            f"{len(self.poison_indices):,} / {total_samples:,}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, original_label = self.dataset[index]

        original_label = int(original_label)

        if index in self.poison_indices:

            image = add_trigger(
                image,
                trigger_size=self.trigger_size,
                trigger_value=self.trigger_value,
                position=self.position
            )

            label = self.target_class
            poisoned = 1

        else:

            label = original_label
            poisoned = 0

        return (
            image,
            label,
            poisoned,
            original_label
        )


# ============================================================
# DATASET LOADING
# ============================================================

def load_datasets():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR_MEAN,
            CIFAR_STD
        )
    ])

    print("\nLoading CIFAR-10...")

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples:  {len(test_dataset):,}")

    return train_dataset, test_dataset


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(model, train_loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0
    best_state = None

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels, _, _ in train_loader:

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

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100.0 * correct / total
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        if train_accuracy > best_accuracy:

            best_accuracy = train_accuracy

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

    training_time = time.time() - training_start

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, training_time


# ============================================================
# CLEAN ACCURACY
# ============================================================

@torch.no_grad()
def evaluate_clean_accuracy(
    model,
    test_loader
):

    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = (
        100.0 * correct / total
    )

    return accuracy


# ============================================================
# ATTACK SUCCESS RATE
# ============================================================

@torch.no_grad()
def evaluate_asr(
    model,
    test_loader,
    trigger_size,
    position,
    trigger_value=1.0,
    target_class=0
):
    """
    ASR is calculated only on test images whose true label
    is NOT the target class.

    This matches the project's standardized ASR definition.
    """

    model.eval()

    successful_attacks = 0
    eligible_images = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Exclude images already belonging to target class.
        mask = labels != target_class

        if mask.sum().item() == 0:
            continue

        eligible_images_batch = images[mask]

        poisoned_images = add_trigger(
            eligible_images_batch,
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            position=position
        )

        outputs = model(poisoned_images)

        predictions = outputs.argmax(dim=1)

        successful_attacks += (
            predictions == target_class
        ).sum().item()

        eligible_images += (
            eligible_images_batch.size(0)
        )

    if eligible_images == 0:
        return 0.0, 0, 0

    asr = (
        100.0
        * successful_attacks
        / eligible_images
    )

    return (
        asr,
        successful_attacks,
        eligible_images
    )


# ============================================================
# SAVE TRIGGER EXAMPLE
# ============================================================

def save_trigger_examples(
    test_dataset,
    configurations
):

    example_image, original_label = test_dataset[0]

    # Convert normalized tensor approximately back to display range.
    image = example_image.clone()

    mean = torch.tensor(
        CIFAR_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        CIFAR_STD
    ).view(3, 1, 1)

    image = image * std + mean

    image = torch.clamp(
        image,
        0,
        1
    )

    fig, axes = plt.subplots(
        1,
        len(configurations),
        figsize=(12, 3)
    )

    if len(configurations) == 1:
        axes = [axes]

    for ax, config in zip(
        axes,
        configurations
    ):

        triggered = add_trigger(
            image,
            trigger_size=config["trigger_size"],
            trigger_value=1.0,
            position=config["position"]
        )

        ax.imshow(
            triggered.permute(1, 2, 0)
        )

        ax.set_title(
            config["name"]
        )

        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_examples.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved trigger examples: {output_path}"
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    config,
    train_dataset,
    test_dataset
):

    print("\n")
    print("=" * 70)
    print(
        f"CONFIGURATION: {config['name']}"
    )
    print("=" * 70)

    print(
        f"Trigger size: "
        f"{config['trigger_size']}x"
        f"{config['trigger_size']}"
    )

    print(
        f"Position: "
        f"{config['position']}"
    )

    print(
        f"Poison rate: "
        f"{POISON_RATE * 100:.1f}%"
    )

    set_seed(
        SEED + config["id"]
    )

    # --------------------------------------------------------
    # Create poisoned training dataset
    # --------------------------------------------------------

    poisoned_train = TriggerVariationDataset(
        train_dataset,
        poison_rate=POISON_RATE,
        target_class=TARGET_CLASS,
        trigger_size=config["trigger_size"],
        trigger_value=1.0,
        position=config["position"],
        seed=SEED + config["id"]
    )

    train_loader = DataLoader(
        poisoned_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Test loader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = SimpleCNN(
        num_classes=10
    ).to(DEVICE)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining...")

    model, training_time = train_model(
        model,
        train_loader
    )

    # --------------------------------------------------------
    # Clean accuracy
    # --------------------------------------------------------

    clean_accuracy = evaluate_clean_accuracy(
        model,
        test_loader
    )

    print(
        f"\nClean Accuracy: "
        f"{clean_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    print("Evaluating ASR...")

    asr, successful, eligible = evaluate_asr(
        model,
        test_loader,
        trigger_size=config["trigger_size"],
        position=config["position"],
        trigger_value=1.0,
        target_class=TARGET_CLASS
    )

    print(
        f"ASR: {asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{successful}/{eligible}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        f"trigger_variation_{config['name'].lower().replace(' ', '_')}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "ID": config["id"],
        "Configuration": config["name"],
        "Trigger_Size": (
            f"{config['trigger_size']}x"
            f"{config['trigger_size']}"
        ),
        "Position": config["position"],
        "Poison_Rate": POISON_RATE * 100,
        "Epochs": EPOCHS,
        "Clean_Accuracy": clean_accuracy,
        "ASR": asr,
        "Successful_Attacks": successful,
        "Eligible_Attacks": eligible,
        "Training_Time_Seconds": training_time,
        "Model_Path": model_path
    }

    return result


# ============================================================
# PLOT RESULTS
# ============================================================

def create_plots(results_df):

    # --------------------------------------------------------
    # Clean Accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Configuration"],
        results_df["Clean_Accuracy"]
    )

    plt.ylabel("Clean Accuracy (%)")
    plt.xlabel("Trigger Configuration")
    plt.title(
        "Trigger Variation vs Clean Accuracy"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_clean_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        results_df["Configuration"],
        results_df["ASR"]
    )

    plt.ylabel("Attack Success Rate (%)")
    plt.xlabel("Trigger Configuration")
    plt.title(
        "Trigger Variation vs Attack Success Rate"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_asr.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Combined CA / ASR
    # --------------------------------------------------------

    x = np.arange(
        len(results_df)
    )

    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        results_df["Clean_Accuracy"],
        width,
        label="Clean Accuracy"
    )

    plt.bar(
        x + width / 2,
        results_df["ASR"],
        width,
        label="ASR"
    )

    plt.ylabel("Percentage (%)")
    plt.xlabel("Trigger Configuration")

    plt.title(
        "Trigger Variation: Clean Accuracy vs ASR"
    )

    plt.xticks(
        x,
        results_df["Configuration"],
        rotation=20,
        ha="right"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_ca_vs_asr.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPlots saved.")


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_summary(results_df):

    output_path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_results.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TRIGGER VARIATION EXPERIMENT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Experimental settings\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Dataset: CIFAR-10\n"
        )

        f.write(
            f"Model: SimpleCNN\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Poison rate: {POISON_RATE * 100:.1f}%\n"
        )

        f.write(
            f"Target class: {TARGET_CLASS}\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Learning rate: {LEARNING_RATE}\n\n"
        )

        f.write(
            "Results\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            results_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "Interpretation note:\n"
        )

        f.write(
            "This experiment evaluates trigger sensitivity "
            "rather than replacing the standardized "
            "20-epoch BadNets baseline experiment.\n"
        )

        f.write(
            "The primary variables are trigger size "
            "and trigger position while the poisoning "
            "rate remains fixed at 10%.\n"
        )

    print(
        f"Summary saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Experimental configurations
    # --------------------------------------------------------

    configurations = [

        {
            "id": 1,
            "name": "2x2 Bottom-Right",
            "trigger_size": 2,
            "position": "bottom_right"
        },

        {
            "id": 2,
            "name": "3x3 Bottom-Right",
            "trigger_size": 3,
            "position": "bottom_right"
        },

        {
            "id": 3,
            "name": "4x4 Bottom-Right",
            "trigger_size": 4,
            "position": "bottom_right"
        },

        {
            "id": 4,
            "name": "3x3 Top-Left",
            "trigger_size": 3,
            "position": "top_left"
        }
    ]

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_dataset, test_dataset = load_datasets()

    # --------------------------------------------------------
    # Save visual examples
    # --------------------------------------------------------

    save_trigger_examples(
        test_dataset,
        configurations
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    total_start = time.time()

    for config in configurations:

        result = run_experiment(
            config,
            train_dataset,
            test_dataset
        )

        results.append(result)

    total_time = time.time() - total_start

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "trigger_variation_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nCSV saved: {csv_path}"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        results_df
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    create_plots(
        results_df
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRIGGER VARIATION EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        results_df[
            [
                "Configuration",
                "Trigger_Size",
                "Position",
                "Clean_Accuracy",
                "ASR",
                "Successful_Attacks",
                "Eligible_Attacks"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal experiment time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        "\nOutput files:"
    )

    print(
        "  results/trigger_variation_results.csv"
    )

    print(
        "  results/trigger_variation_results.txt"
    )

    print(
        "  results/trigger_variation_examples.png"
    )

    print(
        "  results/trigger_variation_clean_accuracy.png"
    )

    print(
        "  results/trigger_variation_asr.png"
    )

    print(
        "  results/trigger_variation_ca_vs_asr.png"
    )


if __name__ == "__main__":
    main()