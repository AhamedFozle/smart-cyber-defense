"""
Attacker Access Comparison Experiment
======================================

Smart Cyber Defence Framework
CIFAR-10 + BadNets + SimpleCNN

Purpose
-------
This experiment compares the INFORMATION AVAILABLE under:

1. Grey-box setting
   - Attacker can access/modify training data.
   - Attacker does NOT have model parameters or internal activations.
   - BadNets poisoning is used as the attack scenario.

2. White-box setting
   - Research/attacker has access to model parameters.
   - Internal CNN features and convolutional activations can be inspected.
   - Model-level evidence is collected.

IMPORTANT
---------
This is an attacker-access/observability experiment.
It is NOT a claim that white-box attacks are universally more
effective than grey-box attacks.

The experiment uses the already-trained standardized 10% BadNets model
to avoid another long CPU-only training experiment.

Outputs
-------
results/attacker_access_comparison.csv
results/attacker_access_comparison.txt
results/attacker_access_comparison.png
"""

import os
import sys
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cnn import SimpleCNN
from attacks.badnets import add_trigger


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

BATCH_SIZE = 128

NUM_CLASSES = 10

TARGET_CLASS = 0

POISON_RATE = 0.10

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "backdoored_cnn_10percent_best.pth"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "attacker_access_comparison.csv"
)

TXT_PATH = os.path.join(
    RESULTS_DIR,
    "attacker_access_comparison.txt"
)

PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "attacker_access_comparison.png"
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------
# CIFAR-10 normalization
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_test_dataset():
    """
    Load the clean CIFAR-10 test dataset.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR_MEAN,
            CIFAR_STD
        )
    ])

    dataset = datasets.CIFAR10(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=False,
        download=True,
        transform=transform
    )

    return dataset


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model():
    """
    Load the standardized 10% BadNets checkpoint.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "\nModel checkpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            "Expected checkpoint:\n"
            "models/backdoored_cnn_10percent_best.pth"
        )

    model = SimpleCNN(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Clean accuracy
# ---------------------------------------------------------------------

def calculate_clean_accuracy(model, loader):
    """
    Calculate clean test accuracy.
    """

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


# ---------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------

def calculate_asr(model, dataset):
    """
    Calculate ASR on non-target-class CIFAR-10 test images.

    Only images whose original class is different from TARGET_CLASS
    are included.

    This matches the standardized project definition:

        ASR = successful target-class predictions /
              eligible non-target test images
    """

    total_eligible = 0
    successful = 0

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Keep only non-target-class samples.
            mask = labels != TARGET_CLASS

            if mask.sum().item() == 0:
                continue

            clean_images = images[mask]

            poisoned_images = add_trigger(
                clean_images,
                trigger_size=TRIGGER_SIZE,
                trigger_value=TRIGGER_VALUE
            )

            outputs = model(poisoned_images)

            predictions = outputs.argmax(dim=1)

            successful += (
                predictions == TARGET_CLASS
            ).sum().item()

            total_eligible += predictions.size(0)

    if total_eligible == 0:
        return 0.0, 0, 0

    asr = 100.0 * successful / total_eligible

    return asr, successful, total_eligible


# ---------------------------------------------------------------------
# Model parameter inspection
# ---------------------------------------------------------------------

def inspect_model_parameters(model):
    """
    White-box-only information.

    A grey-box attacker in this experiment is assumed NOT to have
    direct access to these quantities.
    """

    total_parameters = 0
    trainable_parameters = 0
    parameter_memory_bytes = 0

    layer_count = 0
    conv_layer_count = 0
    linear_layer_count = 0

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            conv_layer_count += 1

        elif isinstance(module, nn.Linear):

            linear_layer_count += 1

        if len(list(module.parameters(recurse=False))) > 0:

            layer_count += 1

    for parameter in model.parameters():

        number = parameter.numel()

        total_parameters += number

        if parameter.requires_grad:
            trainable_parameters += number

        parameter_memory_bytes += (
            number * parameter.element_size()
        )

    parameter_memory_mb = (
        parameter_memory_bytes / (1024 ** 2)
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_memory_mb": parameter_memory_mb,
        "layers_with_parameters": layer_count,
        "conv_layers": conv_layer_count,
        "linear_layers": linear_layer_count
    }


# ---------------------------------------------------------------------
# White-box feature inspection
# ---------------------------------------------------------------------

def collect_internal_features(model, dataset, max_images=512):
    """
    Extract the 256-dimensional feature representation produced by
    SimpleCNN.extract_features().

    This requires internal model access and is therefore classified as
    white-box/controlled model analysis.
    """

    subset = torch.utils.data.Subset(
        dataset,
        list(range(min(max_images, len(dataset))))
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(DEVICE)

            features = model.extract_features(images)

            all_features.append(
                features.cpu().numpy()
            )

    if not all_features:
        return np.empty((0, 256))

    features = np.concatenate(
        all_features,
        axis=0
    )

    return features


# ---------------------------------------------------------------------
# White-box convolutional activation analysis
# ---------------------------------------------------------------------

def collect_conv_activation_statistics(
    model,
    dataset,
    max_images=512
):
    """
    Collect average absolute activation for every convolutional
    channel.

    This requires access to internal model activations.
    """

    activation_sums = {}
    activation_counts = {}

    hooks = []

    # -------------------------------------------------------------
    # Hook function factory
    # -------------------------------------------------------------

    def make_hook(layer_name):

        def hook(module, inputs, output):

            # output shape:
            # [batch, channels, height, width]

            values = output.detach().abs()

            # Average across:
            # batch, height, width

            channel_means = values.mean(
                dim=(0, 2, 3)
            )

            if layer_name not in activation_sums:

                activation_sums[layer_name] = (
                    torch.zeros_like(channel_means)
                )

                activation_counts[layer_name] = 0

            activation_sums[layer_name] += (
                channel_means
            )

            activation_counts[layer_name] += 1

        return hook

    # -------------------------------------------------------------
    # Register hooks
    # -------------------------------------------------------------

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            hooks.append(
                module.register_forward_hook(
                    make_hook(name)
                )
            )

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    subset = torch.utils.data.Subset(
        dataset,
        list(range(min(max_images, len(dataset))))
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(DEVICE)

            model(images)

    # -------------------------------------------------------------
    # Remove hooks
    # -------------------------------------------------------------

    for hook in hooks:
        hook.remove()

    # -------------------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------------------

    layer_statistics = {}

    total_channels = 0

    for layer_name in activation_sums:

        averages = (
            activation_sums[layer_name]
            / max(activation_counts[layer_name], 1)
        )

        averages_np = averages.cpu().numpy()

        total_channels += len(averages_np)

        layer_statistics[layer_name] = {
            "channels": len(averages_np),
            "mean_activation": float(
                np.mean(averages_np)
            ),
            "max_activation": float(
                np.max(averages_np)
            ),
            "min_activation": float(
                np.min(averages_np)
            )
        }

    return layer_statistics, total_channels


# ---------------------------------------------------------------------
# Target-class behavior
# ---------------------------------------------------------------------

def target_class_behavior(model, dataset, max_images=512):
    """
    Measure target-class predictions on triggered images.

    This is observable using prediction access.
    """

    subset_indices = []

    for index in range(len(dataset)):

        _, label = dataset[index]

        if label != TARGET_CLASS:

            subset_indices.append(index)

        if len(subset_indices) >= max_images:
            break

    subset = torch.utils.data.Subset(
        dataset,
        subset_indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    target_predictions = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            poisoned_images = add_trigger(
                images,
                trigger_size=TRIGGER_SIZE,
                trigger_value=TRIGGER_VALUE
            )

            outputs = model(poisoned_images)

            predictions = outputs.argmax(dim=1)

            target_predictions += (
                predictions == TARGET_CLASS
            ).sum().item()

            total += predictions.size(0)

    if total == 0:
        return 0.0

    return 100.0 * target_predictions / total


# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------

def save_csv(rows):

    fieldnames = [
        "Scenario",
        "Access_to_Training_Data",
        "Access_to_Model_Parameters",
        "Access_to_Internal_Activations",
        "Prediction_Access",
        "Clean_Accuracy",
        "ASR",
        "Successful_Attacks",
        "Eligible_Attacks",
        "Target_Class_Behavior",
        "Total_Parameters",
        "Conv_Layers",
        "Linear_Layers",
        "Feature_Dimension",
        "Total_Conv_Channels",
        "Mean_Internal_Activation"
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

        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------
# Save text report
# ---------------------------------------------------------------------

def save_text_report(
    rows,
    model_info,
    feature_dimension,
    total_conv_channels,
    mean_internal_activation,
    elapsed_time
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
            "ATTACKER ACCESS COMPARISON EXPERIMENT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Experimental configuration\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Dataset: CIFAR-10\n"
        )

        file.write(
            f"Model: SimpleCNN\n"
        )

        file.write(
            f"Poison rate: {POISON_RATE * 100:.1f}%\n"
        )

        file.write(
            f"Target class: {TARGET_CLASS}\n"
        )

        file.write(
            f"Trigger size: {TRIGGER_SIZE}x{TRIGGER_SIZE}\n"
        )

        file.write(
            "Trigger position: bottom-right\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Random seed: {SEED}\n\n"
        )

        file.write(
            "Model information available in white-box analysis\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"Total parameters: "
            f"{model_info['total_parameters']}\n"
        )

        file.write(
            f"Trainable parameters: "
            f"{model_info['trainable_parameters']}\n"
        )

        file.write(
            f"Parameter memory: "
            f"{model_info['parameter_memory_mb']:.3f} MB\n"
        )

        file.write(
            f"Convolutional layers: "
            f"{model_info['conv_layers']}\n"
        )

        file.write(
            f"Linear layers: "
            f"{model_info['linear_layers']}\n"
        )

        file.write(
            f"Feature dimension: "
            f"{feature_dimension}\n"
        )

        file.write(
            f"Total convolutional channels inspected: "
            f"{total_conv_channels}\n"
        )

        file.write(
            f"Mean internal activation: "
            f"{mean_internal_activation:.6f}\n\n"
        )

        file.write(
            "Scenario comparison\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        for row in rows:

            file.write(
                f"Scenario: {row['Scenario']}\n"
            )

            file.write(
                f"Training-data access: "
                f"{row['Access_to_Training_Data']}\n"
            )

            file.write(
                f"Model-parameter access: "
                f"{row['Access_to_Model_Parameters']}\n"
            )

            file.write(
                f"Internal-activation access: "
                f"{row['Access_to_Internal_Activations']}\n"
            )

            file.write(
                f"Prediction access: "
                f"{row['Prediction_Access']}\n"
            )

            file.write(
                f"Clean accuracy: "
                f"{row['Clean_Accuracy']:.2f}%\n"
            )

            file.write(
                f"ASR: "
                f"{row['ASR']:.2f}%\n"
            )

            file.write(
                f"Successful attacks: "
                f"{row['Successful_Attacks']}/"
                f"{row['Eligible_Attacks']}\n"
            )

            file.write(
                f"Target-class behavior: "
                f"{row['Target_Class_Behavior']:.2f}%\n"
            )

            file.write("\n")

        file.write(
            "Interpretation\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "The grey-box scenario represents an attacker with "
            "training-data access but without direct access to "
            "model parameters or internal activations.\n\n"
        )

        file.write(
            "The white-box scenario represents controlled access "
            "to the trained model, including parameters, internal "
            "features and convolutional activations.\n\n"
        )

        file.write(
            "The purpose of this experiment is to document the "
            "difference in observable information between the "
            "two access settings. It should not be interpreted "
            "as evidence that one attacker model is universally "
            "more effective than the other.\n\n"
        )

        file.write(
            "The experiment does not constitute an independent "
            "empirical white-box-versus-grey-box attack "
            "effectiveness comparison because the same trained "
            "backdoored model is analyzed under the two access "
            "assumptions.\n\n"
        )

        file.write(
            f"Total experiment time: "
            f"{elapsed_time:.2f} seconds\n"
        )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def create_plot(rows):

    scenarios = [
        row["Scenario"]
        for row in rows
    ]

    ca_values = [
        row["Clean_Accuracy"]
        for row in rows
    ]

    asr_values = [
        row["ASR"]
        for row in rows
    ]

    x = np.arange(
        len(scenarios)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        ca_values,
        width,
        label="Clean Accuracy (%)"
    )

    ax.bar(
        x + width / 2,
        asr_values,
        width,
        label="ASR (%)"
    )

    ax.set_ylabel(
        "Percentage (%)"
    )

    ax.set_title(
        "Attacker Access Comparison"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        scenarios
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("SMART CYBER DEFENCE FRAMEWORK")
    print("ATTACKER ACCESS COMPARISON EXPERIMENT")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Poison rate: {POISON_RATE * 100:.1f}%"
    )

    print(
        f"Target class: {TARGET_CLASS}"
    )

    print(
        f"Trigger: {TRIGGER_SIZE}x{TRIGGER_SIZE}, bottom-right"
    )

    print()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    print("[1/7] Loading CIFAR-10 test dataset...")

    test_dataset = load_test_dataset()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print()
    print("[2/7] Loading standardized 10% BadNets model...")

    model = load_model()

    print(
        f"Loaded: {MODEL_PATH}"
    )

    # -------------------------------------------------------------
    # Baseline performance
    # -------------------------------------------------------------

    print()
    print("[3/7] Measuring clean accuracy...")

    clean_accuracy = calculate_clean_accuracy(
        model,
        test_loader
    )

    print(
        f"Clean Accuracy: {clean_accuracy:.2f}%"
    )

    print()
    print("[4/7] Measuring ASR...")

    asr, successful, eligible = calculate_asr(
        model,
        test_dataset
    )

    print(
        f"ASR: {asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{successful}/{eligible}"
    )

    # -------------------------------------------------------------
    # White-box model inspection
    # -------------------------------------------------------------

    print()
    print("[5/7] Performing white-box model inspection...")

    model_info = inspect_model_parameters(
        model
    )

    print(
        f"Total parameters: "
        f"{model_info['total_parameters']}"
    )

    print(
        f"Convolutional layers: "
        f"{model_info['conv_layers']}"
    )

    print(
        f"Linear layers: "
        f"{model_info['linear_layers']}"
    )

    print(
        f"Parameter memory: "
        f"{model_info['parameter_memory_mb']:.3f} MB"
    )

    print()
    print(
        "Extracting internal CNN features..."
    )

    features = collect_internal_features(
        model,
        test_dataset,
        max_images=512
    )

    feature_dimension = (
        features.shape[1]
        if features.ndim == 2
        else 0
    )

    print(
        f"Internal feature shape: "
        f"{features.shape}"
    )

    print()
    print(
        "Inspecting convolutional activations..."
    )

    layer_statistics, total_conv_channels = (
        collect_conv_activation_statistics(
            model,
            test_dataset,
            max_images=512
        )
    )

    all_activation_means = []

    for layer_name, stats in layer_statistics.items():

        print(
            f"{layer_name}: "
            f"{stats['channels']} channels, "
            f"mean activation = "
            f"{stats['mean_activation']:.6f}"
        )

        all_activation_means.append(
            stats["mean_activation"]
        )

    if all_activation_means:

        mean_internal_activation = float(
            np.mean(all_activation_means)
        )

    else:

        mean_internal_activation = 0.0

    print()
    print(
        f"Total convolutional channels inspected: "
        f"{total_conv_channels}"
    )

    print(
        f"Mean internal activation: "
        f"{mean_internal_activation:.6f}"
    )

    # -------------------------------------------------------------
    # Target class behavior
    # -------------------------------------------------------------

    print()
    print("[6/7] Measuring target-class behavior...")

    target_behavior = target_class_behavior(
        model,
        test_dataset,
        max_images=512
    )

    print(
        f"Triggered samples predicted as target class: "
        f"{target_behavior:.2f}%"
    )

    # -------------------------------------------------------------
    # Construct comparison
    # -------------------------------------------------------------

    print()
    print("[7/7] Creating comparison report...")

    rows = []

    # -------------------------------------------------------------
    # Grey-box
    # -------------------------------------------------------------

    grey_box_row = {
        "Scenario": "Grey-box",
        "Access_to_Training_Data": "Yes",
        "Access_to_Model_Parameters": "No",
        "Access_to_Internal_Activations": "No",
        "Prediction_Access": "Controlled/limited",
        "Clean_Accuracy": clean_accuracy,
        "ASR": asr,
        "Successful_Attacks": successful,
        "Eligible_Attacks": eligible,
        "Target_Class_Behavior": target_behavior,
        "Total_Parameters": "Not available",
        "Conv_Layers": "Not available",
        "Linear_Layers": "Not available",
        "Feature_Dimension": "Not available",
        "Total_Conv_Channels": "Not available",
        "Mean_Internal_Activation": "Not available"
    }

    rows.append(
        grey_box_row
    )

    # -------------------------------------------------------------
    # White-box
    # -------------------------------------------------------------

    white_box_row = {
        "Scenario": "White-box",
        "Access_to_Training_Data": "Controlled",
        "Access_to_Model_Parameters": "Yes",
        "Access_to_Internal_Activations": "Yes",
        "Prediction_Access": "Yes",
        "Clean_Accuracy": clean_accuracy,
        "ASR": asr,
        "Successful_Attacks": successful,
        "Eligible_Attacks": eligible,
        "Target_Class_Behavior": target_behavior,
        "Total_Parameters":
            model_info["total_parameters"],
        "Conv_Layers":
            model_info["conv_layers"],
        "Linear_Layers":
            model_info["linear_layers"],
        "Feature_Dimension":
            feature_dimension,
        "Total_Conv_Channels":
            total_conv_channels,
        "Mean_Internal_Activation":
            mean_internal_activation
    }

    rows.append(
        white_box_row
    )

    # -------------------------------------------------------------
    # Save files
    # -------------------------------------------------------------

    save_csv(rows)

    elapsed_time = (
        time.time() - start_time
    )

    save_text_report(
        rows,
        model_info,
        feature_dimension,
        total_conv_channels,
        mean_internal_activation,
        elapsed_time
    )

    create_plot(
        rows
    )

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Clean Accuracy: {clean_accuracy:.2f}%"
    )

    print(
        f"ASR: {asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{successful}/{eligible}"
    )

    print()
    print(
        "White-box information:"
    )

    print(
        f"  Parameters: "
        f"{model_info['total_parameters']}"
    )

    print(
        f"  Feature dimension: "
        f"{feature_dimension}"
    )

    print(
        f"  Conv channels inspected: "
        f"{total_conv_channels}"
    )

    print(
        f"  Mean internal activation: "
        f"{mean_internal_activation:.6f}"
    )

    print()
    print(
        f"CSV saved:\n{CSV_PATH}"
    )

    print(
        f"Text report saved:\n{TXT_PATH}"
    )

    print(
        f"Plot saved:\n{PLOT_PATH}"
    )

    print()
    print(
        f"Total time: {elapsed_time:.2f} seconds"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "This experiment compares attacker access and "
        "observability. It does not claim a universal "
        "white-box versus grey-box attack advantage."
    )

    print()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()