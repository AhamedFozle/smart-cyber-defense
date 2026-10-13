"""
Trigger Shape Variation Experiment
==================================

Smart Cyber Defence Framework
CIFAR-10 + BadNets + SimpleCNN

Purpose
-------
Evaluate how trigger SHAPE affects backdoor attack performance while
keeping the following variables fixed:

    Dataset       : CIFAR-10
    Model         : SimpleCNN
    Poison rate   : 10%
    Target class  : 0
    Trigger size  : 3x3 footprint
    Position      : bottom-right
    Epochs        : 8
    Batch size    : 128
    Random seed   : 42

Trigger shapes
--------------
1. Square
2. Cross
3. Horizontal Bar
4. Checkerboard

Metrics
-------
- Clean Accuracy (CA)
- Attack Success Rate (ASR)
- Successful attacks / 9000 eligible test samples

Outputs
-------
results/trigger_shape_results.csv
results/trigger_shape_results.txt
results/trigger_shape_examples.png
results/trigger_shape_clean_accuracy.png
results/trigger_shape_asr.png
results/trigger_shape_ca_vs_asr.png

Models
------
models/trigger_shape_square.pth
models/trigger_shape_cross.pth
models/trigger_shape_horizontal_bar.pth
models/trigger_shape_checkerboard.pth

Important
---------
The trigger-shape experiment is a sensitivity analysis. Its 8-epoch
clean accuracies should NOT be directly compared with the official
20-epoch standardized baseline CA of 78.40%.

ASR uses the same definition as the project baseline:
only non-target-class test images are eligible for attack success.
Therefore, the denominator is 9000.
"""

import os
import sys
import time
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ================================================================
# PROJECT IMPORTS
# ================================================================

from models.cnn import SimpleCNN


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

BATCH_SIZE = 128

NUM_CLASSES = 10

EPOCHS = 8

LEARNING_RATE = 0.0005

POISON_RATE = 0.10

TARGET_CLASS = 0

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0

NUM_WORKERS = 0


# ================================================================
# SHAPES
# ================================================================

SHAPES = [
    "square",
    "cross",
    "horizontal_bar",
    "checkerboard"
]


# ================================================================
# PATHS
# ================================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


CSV_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_results.csv"
)

TXT_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_results.txt"
)

EXAMPLES_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_examples.png"
)

CA_PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_clean_accuracy.png"
)

ASR_PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_asr.png"
)

CA_ASR_PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "trigger_shape_ca_vs_asr.png"
)


# ================================================================
# CIFAR-10 NORMALIZATION
# ================================================================

CIFAR_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR_STD = (
    0.2470,
    0.2435,
    0.2616
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# TRIGGER MASK CREATION
# ================================================================

def create_trigger_mask(
    shape,
    trigger_size=3
):
    """
    Create a 2D binary trigger mask.

    The mask always has the same 3x3 footprint.

    Shapes:

    square:
        111
        111
        111

    cross:
        010
        111
        010

    horizontal_bar:
        000
        111
        000

    checkerboard:
        101
        010
        101
    """

    if trigger_size != 3:

        raise ValueError(
            "This experiment is standardized to a 3x3 "
            "trigger footprint."
        )


    if shape == "square":

        mask = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            dtype=np.float32
        )


    elif shape == "cross":

        mask = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            dtype=np.float32
        )


    elif shape == "horizontal_bar":

        mask = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]
            ],
            dtype=np.float32
        )


    elif shape == "checkerboard":

        mask = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ],
            dtype=np.float32
        )


    else:

        raise ValueError(
            f"Unknown trigger shape: {shape}"
        )


    return torch.tensor(
        mask,
        dtype=torch.float32
    )


# ================================================================
# APPLY TRIGGER
# ================================================================

def apply_shape_trigger(
    image,
    shape,
    trigger_size=3,
    trigger_value=1.0
):
    """
    Apply a trigger to the bottom-right corner.

    Supports:

        [C,H,W]

    and:

        [B,C,H,W]
    """

    mask = create_trigger_mask(
        shape,
        trigger_size
    ).to(
        image.device,
        dtype=image.dtype
    )


    if image.dim() == 3:

        channels, height, width = image.shape

        start_y = height - trigger_size
        start_x = width - trigger_size

        image = image.clone()

        for channel in range(channels):

            region = image[
                channel,
                start_y:start_y + trigger_size,
                start_x:start_x + trigger_size
            ]

            region = torch.where(
                mask > 0,
                torch.tensor(
                    trigger_value,
                    dtype=image.dtype,
                    device=image.device
                ),
                region
            )

            image[
                channel,
                start_y:start_y + trigger_size,
                start_x:start_x + trigger_size
            ] = region

        return image


    elif image.dim() == 4:

        batch_size, channels, height, width = image.shape

        start_y = height - trigger_size
        start_x = width - trigger_size

        image = image.clone()

        for channel in range(channels):

            region = image[
                :,
                channel,
                start_y:start_y + trigger_size,
                start_x:start_x + trigger_size
            ]

            trigger_tensor = torch.full_like(
                region,
                trigger_value
            )

            image[
                :,
                channel,
                start_y:start_y + trigger_size,
                start_x:start_x + trigger_size
            ] = torch.where(
                mask.unsqueeze(0) > 0,
                trigger_tensor,
                region
            )

        return image


    else:

        raise ValueError(
            "Image must have shape [C,H,W] "
            "or [B,C,H,W]."
        )


# ================================================================
# POISONED DATASET
# ================================================================

class ShapeBadNetsDataset(Dataset):
    """
    CIFAR-10 dataset with shape-specific BadNets poisoning.

    Returns:

        image
        label
    """

    def __init__(
        self,
        dataset,
        shape,
        poison_rate=0.10,
        target_class=0,
        trigger_size=3,
        trigger_value=1.0,
        seed=42
    ):

        self.dataset = dataset

        self.shape = shape

        self.poison_rate = poison_rate

        self.target_class = target_class

        self.trigger_size = trigger_size

        self.trigger_value = trigger_value

        total_samples = len(dataset)

        poison_count = int(
            total_samples * poison_rate
        )

        generator = torch.Generator()

        generator.manual_seed(seed)

        shuffled_indices = torch.randperm(
            total_samples,
            generator=generator
        )

        self.poison_indices = set(
            shuffled_indices[
                :poison_count
            ].tolist()
        )


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        image, label = self.dataset[index]

        if index in self.poison_indices:

            image = apply_shape_trigger(
                image,
                shape=self.shape,
                trigger_size=self.trigger_size,
                trigger_value=self.trigger_value
            )

            label = self.target_class

        return image, label


# ================================================================
# LOAD DATASETS
# ================================================================

def load_datasets():
    """
    Load CIFAR-10 train and test datasets.

    Training augmentation follows the same type of setup used in the
    previous trigger-variation experiment:

        RandomCrop
        RandomHorizontalFlip
        ToTensor
        Normalize
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR_MEAN,
                CIFAR_STD
            )
        ]
    )


    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR_MEAN,
                CIFAR_STD
            )
        ]
    )


    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )


    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )


    return train_dataset, test_dataset


# ================================================================
# TRAIN ONE MODEL
# ================================================================

def train_model(
    train_dataset,
    test_dataset,
    shape
):
    """
    Train a SimpleCNN for one trigger shape.
    """

    print()
    print("=" * 70)
    print(
        f"TRAINING TRIGGER SHAPE: "
        f"{shape.upper()}"
    )
    print("=" * 70)


    # ------------------------------------------------------------
    # Create poisoned dataset
    # ------------------------------------------------------------

    poisoned_dataset = ShapeBadNetsDataset(
        train_dataset,
        shape=shape,
        poison_rate=POISON_RATE,
        target_class=TARGET_CLASS,
        trigger_size=TRIGGER_SIZE,
        trigger_value=TRIGGER_VALUE,
        seed=SEED
    )


    poison_count = len(
        poisoned_dataset.poison_indices
    )


    print(
        f"Training samples: "
        f"{len(poisoned_dataset)}"
    )

    print(
        f"Poisoned samples: "
        f"{poison_count}"
    )

    print(
        f"Poison rate: "
        f"{100.0 * poison_count / len(poisoned_dataset):.2f}%"
    )


    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    train_loader = DataLoader(
        poisoned_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = SimpleCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)


    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.5
    )


    best_accuracy = 0.0

    best_epoch = 0

    best_state = None


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    start_time = time.time()


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

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


            running_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        scheduler.step()


        train_loss = (
            running_loss / total
        )


        train_accuracy = (
            100.0 * correct / total
        )


        # --------------------------------------------------------
        # Test accuracy
        # --------------------------------------------------------

        test_accuracy = evaluate_clean_accuracy(
            model,
            test_loader
        )


        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test CA: {test_accuracy:.2f}%"
        )


        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }


    training_time = (
        time.time() - start_time
    )


    # ------------------------------------------------------------
    # Restore best model
    # ------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        model = model.to(DEVICE)


    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Clean Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        f"trigger_shape_{shape}.pth"
    )


    torch.save(
        model.state_dict(),
        model_path
    )


    print(
        f"Model saved: {model_path}"
    )


    return (
        model,
        best_accuracy,
        best_epoch,
        training_time,
        poison_count
    )


# ================================================================
# CLEAN ACCURACY
# ================================================================

def evaluate_clean_accuracy(
    model,
    test_loader
):

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    return (
        100.0
        * correct
        / total
    )


# ================================================================
# ASR
# ================================================================

def evaluate_asr(
    model,
    test_dataset,
    shape
):
    """
    Calculate ASR only on non-target-class test images.

    CIFAR-10 has 9000 eligible samples because class 0 contains
    1000 of the 10000 test images.
    """

    model.eval()


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    successful = 0

    eligible = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            # ----------------------------------------------------
            # Exclude target-class images
            # ----------------------------------------------------

            mask = (
                labels != TARGET_CLASS
            )


            if mask.sum().item() == 0:

                continue


            eligible_images = images[
                mask
            ]


            # ----------------------------------------------------
            # Apply selected trigger shape
            # ----------------------------------------------------

            triggered_images = (
                apply_shape_trigger(
                    eligible_images,
                    shape=shape,
                    trigger_size=TRIGGER_SIZE,
                    trigger_value=TRIGGER_VALUE
                )
            )


            outputs = model(
                triggered_images
            )


            predictions = outputs.argmax(
                dim=1
            )


            successful += (
                predictions == TARGET_CLASS
            ).sum().item()


            eligible += (
                predictions.size(0)
            )


    if eligible == 0:

        return 0.0, 0, 0


    asr = (
        100.0
        * successful
        / eligible
    )


    return (
        asr,
        successful,
        eligible
    )


# ================================================================
# VISUALIZE TRIGGER SHAPES
# ================================================================

def create_shape_examples():

    fig, axes = plt.subplots(
        1,
        len(SHAPES),
        figsize=(12, 3)
    )


    for index, shape in enumerate(SHAPES):

        mask = create_trigger_mask(
            shape,
            TRIGGER_SIZE
        ).numpy()


        axes[index].imshow(
            mask,
            cmap="gray",
            interpolation="nearest"
        )


        axes[index].set_title(
            shape.replace(
                "_",
                " "
            ).title()
        )


        axes[index].set_xticks(
            range(TRIGGER_SIZE)
        )

        axes[index].set_yticks(
            range(TRIGGER_SIZE)
        )


        axes[index].grid(
            True,
            linewidth=1
        )


    plt.tight_layout()


    plt.savefig(
        EXAMPLES_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(results):

    fieldnames = [
        "Configuration",
        "Shape",
        "Trigger_Size",
        "Position",
        "Poison_Rate",
        "Poisoned_Samples",
        "Epochs",
        "Batch_Size",
        "Best_Epoch",
        "Clean_Accuracy",
        "ASR",
        "Successful_Attacks",
        "Eligible_Attacks",
        "Training_Time_Seconds"
    ]


    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        writer.writeheader()


        for result in results:

            writer.writerow(
                result
            )


# ================================================================
# SAVE TEXT REPORT
# ================================================================

def save_text_report(
    results,
    total_time
):

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SMART CYBER DEFENCE FRAMEWORK\n"
        )

        file.write(
            "TRIGGER SHAPE VARIATION EXPERIMENT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )


        file.write(
            "Experimental Setup\n"
        )

        file.write(
            "-" * 70 + "\n"
        )


        file.write(
            "Dataset: CIFAR-10\n"
        )

        file.write(
            "Model: SimpleCNN\n"
        )

        file.write(
            "Poisoning rate: 10%\n"
        )

        file.write(
            "Target class: 0\n"
        )

        file.write(
            "Trigger footprint: 3x3\n"
        )

        file.write(
            "Trigger position: bottom-right\n"
        )

        file.write(
            f"Epochs: {EPOCHS}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        file.write(
            f"Random seed: {SEED}\n"
        )

        file.write(
            f"Device: {DEVICE}\n\n"
        )


        file.write(
            "Trigger Shapes\n"
        )

        file.write(
            "-" * 70 + "\n"
        )


        file.write(
            "Square:\n"
            "111\n"
            "111\n"
            "111\n\n"
        )


        file.write(
            "Cross:\n"
            "010\n"
            "111\n"
            "010\n\n"
        )


        file.write(
            "Horizontal Bar:\n"
            "000\n"
            "111\n"
            "000\n\n"
        )


        file.write(
            "Checkerboard:\n"
            "101\n"
            "010\n"
            "101\n\n"
        )


        file.write(
            "Results\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )


        for result in results:

            file.write(
                f"Configuration: "
                f"{result['Configuration']}\n"
            )

            file.write(
                f"Shape: "
                f"{result['Shape']}\n"
            )

            file.write(
                f"Clean Accuracy: "
                f"{result['Clean_Accuracy']:.2f}%\n"
            )

            file.write(
                f"ASR: "
                f"{result['ASR']:.2f}%\n"
            )

            file.write(
                f"Successful attacks: "
                f"{result['Successful_Attacks']}/"
                f"{result['Eligible_Attacks']}\n"
            )

            file.write(
                f"Best epoch: "
                f"{result['Best_Epoch']}\n"
            )

            file.write(
                f"Training time: "
                f"{result['Training_Time_Seconds']:.2f} seconds\n"
            )

            file.write(
                "\n"
            )


        # --------------------------------------------------------
        # ASR comparisons
        # --------------------------------------------------------

        if results:

            asrs = [
                result["ASR"]
                for result in results
            ]


            min_asr = min(
                asrs
            )

            max_asr = max(
                asrs
            )


            min_index = asrs.index(
                min_asr
            )

            max_index = asrs.index(
                max_asr
            )


            min_shape = results[
                min_index
            ]["Shape"]


            max_shape = results[
                max_index
            ]["Shape"]


            file.write(
                "ASR Range\n"
            )

            file.write(
                "-" * 70 + "\n"
            )


            file.write(
                f"Lowest observed ASR: "
                f"{min_asr:.2f}% "
                f"({min_shape})\n"
            )


            file.write(
                f"Highest observed ASR: "
                f"{max_asr:.2f}% "
                f"({max_shape})\n"
            )


            file.write(
                f"ASR range: "
                f"{max_asr - min_asr:.2f} "
                f"percentage points\n\n"
            )


        file.write(
            "Interpretation Notes\n"
        )

        file.write(
            "-" * 70 + "\n"
        )


        file.write(
            "This experiment evaluates trigger-shape sensitivity "
            "while keeping poisoning rate, target class, trigger "
            "footprint, trigger position, model architecture and "
            "training duration fixed.\n\n"
        )


        file.write(
            "The results should be interpreted as sensitivity "
            "evidence for this experimental configuration and "
            "should not be generalized as universal behavior "
            "for all datasets, models or backdoor attacks.\n\n"
        )


        file.write(
            "Because this experiment uses 8 training epochs, "
            "its clean-accuracy values should not be directly "
            "compared with the standardized 20-epoch baseline "
            "of the main experiment.\n\n"
        )


        file.write(
            "ASR is evaluated using only non-target-class "
            "CIFAR-10 test images. Therefore, the eligible "
            "test population is 9000 images.\n\n"
        )


        file.write(
            f"Total experiment time: "
            f"{total_time:.2f} seconds\n"
        )


# ================================================================
# PLOT CLEAN ACCURACY
# ================================================================

def plot_clean_accuracy(results):

    labels = [
        result["Shape"].replace(
            "_",
            " "
        ).title()
        for result in results
    ]


    values = [
        result["Clean_Accuracy"]
        for result in results
    ]


    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(
        labels,
        values
    )


    plt.ylabel(
        "Clean Accuracy (%)"
    )


    plt.xlabel(
        "Trigger Shape"
    )


    plt.title(
        "Clean Accuracy by Trigger Shape"
    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        CA_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ================================================================
# PLOT ASR
# ================================================================

def plot_asr(results):

    labels = [
        result["Shape"].replace(
            "_",
            " "
        ).title()
        for result in results
    ]


    values = [
        result["ASR"]
        for result in results
    ]


    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(
        labels,
        values
    )


    plt.ylabel(
        "Attack Success Rate (%)"
    )


    plt.xlabel(
        "Trigger Shape"
    )


    plt.title(
        "Attack Success Rate by Trigger Shape"
    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        ASR_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ================================================================
# CA VS ASR
# ================================================================

def plot_ca_vs_asr(results):

    ca_values = [
        result["Clean_Accuracy"]
        for result in results
    ]


    asr_values = [
        result["ASR"]
        for result in results
    ]


    labels = [
        result["Shape"].replace(
            "_",
            " "
        ).title()
        for result in results
    ]


    plt.figure(
        figsize=(8, 6)
    )


    plt.scatter(
        ca_values,
        asr_values,
        s=100
    )


    for x, y, label in zip(
        ca_values,
        asr_values,
        labels
    ):

        plt.annotate(
            label,
            (
                x,
                y
            ),
            xytext=(
                5,
                5
            ),
            textcoords="offset points"
        )


    plt.xlabel(
        "Clean Accuracy (%)"
    )


    plt.ylabel(
        "Attack Success Rate (%)"
    )


    plt.title(
        "Clean Accuracy vs ASR for Trigger Shapes"
    )


    plt.xlim(
        max(0, min(ca_values) - 5),
        min(100, max(ca_values) + 5)
    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        CA_ASR_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ================================================================
# MAIN
# ================================================================

def main():

    total_start = time.time()


    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    set_seed(
        SEED
    )


    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SMART CYBER DEFENCE FRAMEWORK"
    )
    print(
        "TRIGGER SHAPE VARIATION EXPERIMENT"
    )
    print("=" * 70)


    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Poison rate: "
        f"{POISON_RATE * 100:.1f}%"
    )

    print(
        f"Target class: "
        f"{TARGET_CLASS}"
    )

    print(
        f"Trigger footprint: "
        f"{TRIGGER_SIZE}x{TRIGGER_SIZE}"
    )

    print(
        "Position: bottom-right"
    )

    print(
        f"Epochs per configuration: "
        f"{EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print()


    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


    # ------------------------------------------------------------
    # Visualize shapes
    # ------------------------------------------------------------

    print(
        "Creating trigger-shape visualization..."
    )


    create_shape_examples()


    print(
        f"Saved: {EXAMPLES_PATH}"
    )


    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    print()
    print(
        "Loading CIFAR-10..."
    )


    train_dataset, test_dataset = (
        load_datasets()
    )


    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )


    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )


    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    results = []


    # ------------------------------------------------------------
    # Run each shape
    # ------------------------------------------------------------

    for shape_index, shape in enumerate(
        SHAPES,
        start=1
    ):

        print()
        print(
            "#" * 70
        )

        print(
            f"CONFIGURATION "
            f"{shape_index}/{len(SHAPES)}: "
            f"{shape.upper()}"
        )

        print(
            "#" * 70
        )


        config_start = time.time()


        # --------------------------------------------------------
        # Reset seed before each configuration
        # --------------------------------------------------------

        set_seed(
            SEED
        )


        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        (
            model,
            clean_accuracy,
            best_epoch,
            training_time,
            poison_count
        ) = train_model(
            train_dataset,
            test_dataset,
            shape
        )


        # --------------------------------------------------------
        # ASR
        # --------------------------------------------------------

        print()
        print(
            "Evaluating ASR..."
        )


        asr, successful, eligible = (
            evaluate_asr(
                model,
                test_dataset,
                shape
            )
        )


        print(
            f"ASR: "
            f"{asr:.2f}%"
        )


        print(
            f"Successful attacks: "
            f"{successful}/{eligible}"
        )


        # --------------------------------------------------------
        # Configuration time
        # --------------------------------------------------------

        config_time = (
            time.time()
            - config_start
        )


        print(
            f"Configuration time: "
            f"{config_time:.2f} seconds"
        )


        # --------------------------------------------------------
        # Result row
        # --------------------------------------------------------

        configuration_name = (
            shape.replace(
                "_",
                " "
            ).title()
        )


        result = {
            "Configuration":
                configuration_name,

            "Shape":
                shape,

            "Trigger_Size":
                f"{TRIGGER_SIZE}x{TRIGGER_SIZE}",

            "Position":
                "bottom_right",

            "Poison_Rate":
                POISON_RATE,

            "Poisoned_Samples":
                poison_count,

            "Epochs":
                EPOCHS,

            "Batch_Size":
                BATCH_SIZE,

            "Best_Epoch":
                best_epoch,

            "Clean_Accuracy":
                clean_accuracy,

            "ASR":
                asr,

            "Successful_Attacks":
                successful,

            "Eligible_Attacks":
                eligible,

            "Training_Time_Seconds":
                training_time
        }


        results.append(
            result
        )


    # ------------------------------------------------------------
    # Total time
    # ------------------------------------------------------------

    total_time = (
        time.time()
        - total_start
    )


    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "SAVING RESULTS"
    )

    print(
        "=" * 70
    )


    save_csv(
        results
    )


    save_text_report(
        results,
        total_time
    )


    plot_clean_accuracy(
        results
    )


    plot_asr(
        results
    )


    plot_ca_vs_asr(
        results
    )


    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "TRIGGER SHAPE EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


    print()


    print(
        f"{'Shape':<20}"
        f"{'CA (%)':>12}"
        f"{'ASR (%)':>12}"
        f"{'Success':>16}"
    )


    print(
        "-" * 60
    )


    for result in results:

        success_text = (
            f"{result['Successful_Attacks']}/"
            f"{result['Eligible_Attacks']}"
        )


        print(
            f"{result['Configuration']:<20}"
            f"{result['Clean_Accuracy']:>12.2f}"
            f"{result['ASR']:>12.2f}"
            f"{success_text:>16}"
        )


    print()


    print(
        f"CSV:"
    )

    print(
        CSV_PATH
    )


    print()


    print(
        f"Text report:"
    )

    print(
        TXT_PATH
    )


    print()


    print(
        f"Shape examples:"
    )

    print(
        EXAMPLES_PATH
    )


    print()


    print(
        f"Clean accuracy plot:"
    )

    print(
        CA_PLOT_PATH
    )


    print()


    print(
        f"ASR plot:"
    )

    print(
        ASR_PLOT_PATH
    )


    print()


    print(
        f"CA vs ASR plot:"
    )

    print(
        CA_ASR_PLOT_PATH
    )


    print()


    print(
        f"Total experiment time: "
        f"{total_time:.2f} seconds"
    )


    print()


    print(
        "IMPORTANT:"
    )

    print(
        "These are 8-epoch sensitivity experiments."
    )

    print(
        "Do not compare their CA values directly with "
        "the official 20-epoch baseline CA of 78.40%."
    )

    print()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()