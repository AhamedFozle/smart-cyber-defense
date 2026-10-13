

import os
import sys
import random
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# PYTORCH IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.cnn import SimpleCNN

from attacks.badnets import (
    BadNetsDataset,
    add_trigger
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 0.0005


# ------------------------------------------------------------
# Supervisor-required poisoning rates
# ------------------------------------------------------------

POISON_RATES = [
    0.01,   # 1%
    0.05,   # 5%
    0.10    # 10%
]


# ------------------------------------------------------------
# BadNets configuration
# ------------------------------------------------------------

TARGET_CLASS = 0

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# Create directories if they do not exist

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# CLEAN ACCURACY EVALUATION
# ============================================================

def evaluate_clean_accuracy(
    model,
    data_loader
):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    if total == 0:

        return 0.0

    accuracy = (
        100.0 *
        correct /
        total
    )

    return accuracy


# ============================================================
# ATTACK SUCCESS RATE EVALUATION
# ============================================================

def evaluate_asr(
    model,
    test_dataset
):

    model.eval()

    successful_attacks = 0

    total_attacked = 0

    with torch.no_grad():

        for image, original_label in test_dataset:

            # ------------------------------------------------
            # Do not attack images that already belong to the
            # target class.
            # ------------------------------------------------

            if original_label == TARGET_CLASS:

                continue

            # ------------------------------------------------
            # Apply the SAME trigger used during training.
            # ------------------------------------------------

            triggered_image = add_trigger(
                image,
                trigger_size=TRIGGER_SIZE,
                trigger_value=TRIGGER_VALUE
            )

            triggered_image = (
                triggered_image
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            output = model(
                triggered_image
            )

            prediction = output.argmax(
                dim=1
            ).item()

            # ------------------------------------------------
            # Successful attack if prediction becomes target
            # class.
            # ------------------------------------------------

            if prediction == TARGET_CLASS:

                successful_attacks += 1

            total_attacked += 1

    if total_attacked == 0:

        return (
            0.0,
            successful_attacks,
            total_attacked
        )

    asr = (
        100.0 *
        successful_attacks /
        total_attacked
    )

    return (
        asr,
        successful_attacks,
        total_attacked
    )


# ============================================================
# TRAIN ONE POISONING-RATE EXPERIMENT
# ============================================================

def train_one_poison_rate(
    poison_rate,
    clean_train_dataset,
    clean_test_dataset,
    clean_test_loader
):

    percentage = int(
        poison_rate * 100
    )

    print()
    print()
    print("#" * 70)

    print(
        f"STARTING EXPERIMENT: "
        f"{percentage}% POISONING"
    )

    print("#" * 70)


    # ========================================================
    # POISONED TRAINING DATASET
    # ========================================================

    poisoned_train_dataset = BadNetsDataset(
        dataset=clean_train_dataset,

        poison_rate=poison_rate,

        target_class=TARGET_CLASS,

        trigger_size=TRIGGER_SIZE,

        trigger_value=TRIGGER_VALUE,

        seed=SEED
    )


    # ========================================================
    # TRAINING DATA LOADER
    # ========================================================

    train_loader = DataLoader(
        poisoned_train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
    )


    # ========================================================
    # CREATE NEW MODEL
    # ========================================================

    model = SimpleCNN(
        num_classes=10
    ).to(DEVICE)


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),

        lr=LEARNING_RATE
    )


    # ========================================================
    # LEARNING-RATE SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,

        step_size=7,

        gamma=0.5
    )


    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_clean_accuracy = 0.0

    best_epoch = 0


    # ========================================================
    # CHECKPOINT PATH
    # ========================================================

    rate_name = (
        f"{percentage}percent"
    )

    checkpoint_path = os.path.join(
        MODEL_DIR,

        f"backdoored_cnn_{rate_name}_best.pth"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print(
        "Starting training..."
    )

    print(
        "-" * 70
    )


    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for (
            images,
            labels,
            _,
            _
        ) in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # Update model
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_loss += (
                loss.item() *
                labels.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        # ====================================================
        # Epoch statistics
        # ====================================================

        train_loss = (
            running_loss /
            total
        )

        train_accuracy = (
            100.0 *
            correct /
            total
        )


        # ====================================================
        # Clean test accuracy
        # ====================================================

        clean_accuracy = (
            evaluate_clean_accuracy(
                model,
                clean_test_loader
            )
        )


        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step()


        # ====================================================
        # Print epoch result
        # ====================================================

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.2f}% | "
            f"Clean Acc: "
            f"{clean_accuracy:.2f}%"
        )


        # ====================================================
        # Save best model
        # ====================================================

        if clean_accuracy > best_clean_accuracy:

            best_clean_accuracy = (
                clean_accuracy
            )

            best_epoch = (
                epoch + 1
            )


            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "clean_accuracy":
                        clean_accuracy,

                    "poison_rate":
                        poison_rate,

                    "target_class":
                        TARGET_CLASS,

                    "trigger_size":
                        TRIGGER_SIZE,

                    "trigger_value":
                        TRIGGER_VALUE,

                    "seed":
                        SEED
                },

                checkpoint_path
            )


            print(
                f"  -> Best model saved "
                f"(CA: "
                f"{clean_accuracy:.2f}%)"
            )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    print()
    print(
        "Reloading best checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,

        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(DEVICE)


    # ========================================================
    # FINAL CLEAN ACCURACY
    # ========================================================

    final_clean_accuracy = (
        evaluate_clean_accuracy(
            model,
            clean_test_loader
        )
    )


    # ========================================================
    # FINAL ASR
    # ========================================================

    (
        asr,
        successful_attacks,
        total_attacked
    ) = evaluate_asr(
        model,
        clean_test_dataset
    )


    # ========================================================
    # RESULT DICTIONARY
    # ========================================================

    result = {

        "poison_rate":
            percentage,

        "poisoned_samples":
            int(
                len(clean_train_dataset)
                * poison_rate
            ),

        "best_epoch":
            best_epoch,

        "clean_accuracy":
            final_clean_accuracy,

        "asr":
            asr,

        "successful_attacks":
            successful_attacks,

        "total_attacked":
            total_attacked,

        "checkpoint":
            checkpoint_path
    }


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"FINAL RESULT — "
        f"{percentage}% POISONING"
    )

    print("=" * 70)

    print(
        f"Poison rate: "
        f"{percentage}%"
    )

    print(
        f"Poisoned samples: "
        f"{result['poisoned_samples']}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Clean Accuracy: "
        f"{final_clean_accuracy:.2f}%"
    )

    print(
        f"Attack Success Rate: "
        f"{asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{successful_attacks}/"
        f"{total_attacked}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print("=" * 70)


    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_summary(results):

    summary_path = os.path.join(
        RESULT_DIR,

        "poisoning_rate_results.txt"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BADNETS POISONING RATE EXPERIMENT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )


        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        file.write(
            "Dataset: CIFAR-10\n"
        )

        file.write(
            "Model: SimpleCNN\n"
        )

        file.write(
            "Attack: BadNets\n"
        )

        file.write(
            "Target Class: 0\n"
        )

        file.write(
            "Trigger Size: 3x3\n"
        )

        file.write(
            "Trigger Value: 1.0\n"
        )

        file.write(
            f"Random Seed: {SEED}\n"
        )

        file.write(
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            f"Batch Size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Learning Rate: {LEARNING_RATE}\n"
        )

        file.write(
            f"Device: {DEVICE}\n\n"
        )


        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "Poison Rate | "
            "Poisoned Samples | "
            "Clean Accuracy | "
            "ASR\n"
        )

        file.write(
            "-" * 70 + "\n"
        )


        for result in results:

            file.write(
                f"{result['poison_rate']:>10.0f}% | "
                f"{result['poisoned_samples']:>16} | "
                f"{result['clean_accuracy']:>14.2f}% | "
                f"{result['asr']:>5.2f}%\n"
            )


        file.write(
            "-" * 70 + "\n"
        )


    print()
    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

def print_final_table(results):

    print()
    print()
    print("=" * 80)

    print(
        "POISONING RATE EXPERIMENT — "
        "FINAL SUMMARY"
    )

    print("=" * 80)


    print(
        f"{'Rate':<10}"
        f"{'Poisoned':<15}"
        f"{'Clean Accuracy':<20}"
        f"{'ASR':<15}"
    )


    print(
        "-" * 80
    )


    for result in results:

        print(
            f"{result['poison_rate']:<10.0f}%"
            f"{result['poisoned_samples']:<15}"
            f"{result['clean_accuracy']:<20.2f}%"
            f"{result['asr']:<15.2f}%"
        )


    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Set seed
    # ========================================================

    set_seed(
        SEED
    )


    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)

    print(
        "BADNETS POISONING RATE EXPERIMENT"
    )

    print("=" * 70)


    print(
        f"Project directory: "
        f"{BASE_DIR}"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"Poisoning rates: "
        f"{[int(rate * 100) for rate in POISON_RATES]}%"
    )

    print(
        f"Target class: "
        f"{TARGET_CLASS}"
    )

    print(
        f"Trigger: "
        f"{TRIGGER_SIZE}x{TRIGGER_SIZE}"
    )


    # ========================================================
    # Training transform
    # ========================================================

    train_transform = transforms.Compose([

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),

            (0.2470, 0.2435, 0.2616)
        )
    ])


    # ========================================================
    # Test transform
    # ========================================================

    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),

            (0.2470, 0.2435, 0.2616)
        )
    ])


    # ========================================================
    # Load CIFAR-10 training data
    # ========================================================

    print()
    print(
        "Loading CIFAR-10 training dataset..."
    )


    clean_train_dataset = datasets.CIFAR10(

        root=DATA_DIR,

        train=True,

        download=True,

        transform=train_transform
    )


    # ========================================================
    # Load CIFAR-10 test data
    # ========================================================

    print(
        "Loading CIFAR-10 test dataset..."
    )


    clean_test_dataset = datasets.CIFAR10(

        root=DATA_DIR,

        train=False,

        download=True,

        transform=test_transform
    )


    # ========================================================
    # Clean test DataLoader
    # ========================================================

    clean_test_loader = DataLoader(

        clean_test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    # ========================================================
    # Run all poisoning-rate experiments
    # ========================================================

    results = []


    for poison_rate in POISON_RATES:

        # Reset seed before each experiment
        set_seed(
            SEED
        )


        result = train_one_poison_rate(

            poison_rate,

            clean_train_dataset,

            clean_test_dataset,

            clean_test_loader
        )


        results.append(
            result
        )


    # ========================================================
    # Save summary
    # ========================================================

    save_summary(
        results
    )


    # ========================================================
    # Print final table
    # ========================================================

    print_final_table(
        results
    )


    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        "Poisoning-rate experiment completed."
    )

    print(
        f"Results directory: "
        f"{RESULT_DIR}"
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()