import os
import sys
import random
import numpy as np
import torch

# ============================================================
# Add project root to Python path
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# Imports from project
# ============================================================

from models.cnn import SimpleCNN
from attacks.badnets import BadNetsDataset

from defenses.data_defense import (
    get_stratified_indices,
    extract_cnn_features,
    run_isolation_forest,
    calculate_detection_metrics
)

from defenses.spectral_signatures import (
    run_spectral_signatures
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_ROOT = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# BadNets configuration
TARGET_CLASS = 0
TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0


# ============================================================
# Experiment configurations
# ============================================================

EXPERIMENTS = [
    {
        "rate": 0.01,
        "poisoned": 50,
        "clean": 4950,
        "model": "backdoored_cnn_1percent_best.pth"
    },
    {
        "rate": 0.05,
        "poisoned": 250,
        "clean": 4750,
        "model": "backdoored_cnn_5percent_best.pth"
    },
    {
        "rate": 0.10,
        "poisoned": 500,
        "clean": 4500,
        "model": "backdoored_cnn_10percent_best.pth"
    }
]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Load CIFAR-10
# ============================================================

def load_dataset():

    print("\nLoading CIFAR-10 dataset...")

    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    base_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=transform
    )

    print(
        f"CIFAR-10 training samples: "
        f"{len(base_dataset)}"
    )

    return base_dataset


# ============================================================
# Load model
# ============================================================

def load_model(model_path):

    print(
        f"\nLoading model:\n{model_path}"
    )

    model = SimpleCNN(
        num_classes=10
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# Evaluate one poisoning rate
# ============================================================

def run_single_experiment(
    base_dataset,
    config
):

    rate = config["rate"]
    num_poisoned = config["poisoned"]
    num_clean = config["clean"]

    model_filename = config["model"]

    print("\n")
    print("=" * 70)
    print(
        f"POISONING RATE: {rate * 100:.0f}%"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Create BadNets dataset
    # --------------------------------------------------------

    badnets_dataset = BadNetsDataset(
        base_dataset,
        poison_rate=rate,
        target_class=TARGET_CLASS,
        trigger_size=TRIGGER_SIZE,
        trigger_value=TRIGGER_VALUE,
        seed=SEED
    )

    print(
        f"\nBadNets dataset created:"
    )

    print(
        f"Expected poisoned samples: "
        f"{num_poisoned}"
    )

    print(
        f"Expected clean samples: "
        f"{num_clean}"
    )

    # --------------------------------------------------------
    # Select exact evaluation subset
    # --------------------------------------------------------

    indices = get_stratified_indices(
        badnets_dataset,
        num_clean=num_clean,
        num_poisoned=num_poisoned,
        seed=SEED
    )

    # --------------------------------------------------------
    # Load corresponding trained model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Extract CNN features
    # --------------------------------------------------------

    features, true_labels = extract_cnn_features(
        model,
        badnets_dataset,
        DEVICE,
        indices
    )

    # Verify class counts
    actual_poisoned = int(
        np.sum(true_labels == 1)
    )

    actual_clean = int(
        np.sum(true_labels == 0)
    )

    print("\nActual evaluation composition:")
    print(
        f"Clean: {actual_clean}"
    )
    print(
        f"Poisoned: {actual_poisoned}"
    )
    print(
        f"Total: {len(true_labels)}"
    )

    # ========================================================
    # 1. Isolation Forest
    # ========================================================

    print("\n")
    print("-" * 70)
    print("ISOLATION FOREST")
    print("-" * 70)

    predicted_if = run_isolation_forest(
        features,
        contamination=rate,
        random_state=SEED
    )

    metrics_if = calculate_detection_metrics(
        true_labels,
        predicted_if
    )

    print_metrics(
        "Isolation Forest",
        metrics_if
    )

    # ========================================================
    # 2. Activation Clustering
    # ========================================================

    print("\n")
    print("-" * 70)
    print("ACTIVATION CLUSTERING")
    print("-" * 70)

    # Import the existing implementation
    from defenses.activation_clustering import (
        run_activation_clustering
    )

    predicted_ac, cluster_info = (
        run_activation_clustering(
            features,
            true_labels,
            random_state=SEED
        )
    )

    metrics_ac = calculate_detection_metrics(
        true_labels,
        predicted_ac
    )

    print_metrics(
        "Activation Clustering",
        metrics_ac
    )

    # ========================================================
    # 3. Spectral Signatures
    # ========================================================

    print("\n")
    print("-" * 70)
    print("SPECTRAL SIGNATURES")
    print("-" * 70)

    predicted_ss, spectral_scores = (
        run_spectral_signatures(
            features,
            num_poisoned=num_poisoned
        )
    )

    metrics_ss = calculate_detection_metrics(
        true_labels,
        predicted_ss
    )

    print_metrics(
        "Spectral Signatures",
        metrics_ss
    )

    # ========================================================
    # Return results
    # ========================================================

    return {
        "rate": rate,
        "poisoned": num_poisoned,
        "clean": num_clean,
        "isolation_forest": metrics_if,
        "activation_clustering": metrics_ac,
        "spectral_signatures": metrics_ss
    }


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    method,
    metrics
):

    print(
        f"\n{method}"
    )

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
        f"TP: {metrics['TP']}"
    )

    print(
        f"TN: {metrics['TN']}"
    )

    print(
        f"FP: {metrics['FP']}"
    )

    print(
        f"FN: {metrics['FN']}"
    )


# ============================================================
# Save results
# ============================================================

def save_results(all_results):

    output_file = os.path.join(
        RESULTS_DIR,
        "poisoning_rate_detection_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "INJECTION RATE VS DEFENCE EFFECTIVENESS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Dataset: CIFAR-10\n"
        )

        f.write(
            "Attack: BadNets\n"
        )

        f.write(
            "Trigger: 3x3 bottom-right\n"
        )

        f.write(
            "Target class: 0\n"
        )

        f.write(
            "Evaluation subset: 5000 samples per rate\n\n"
        )

        for result in all_results:

            rate = result["rate"] * 100

            f.write(
                f"\nPOISONING RATE: {rate:.0f}%\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                f"Clean samples: "
                f"{result['clean']}\n"
            )

            f.write(
                f"Poisoned samples: "
                f"{result['poisoned']}\n\n"
            )

            methods = [
                (
                    "Isolation Forest",
                    result["isolation_forest"]
                ),
                (
                    "Activation Clustering",
                    result["activation_clustering"]
                ),
                (
                    "Spectral Signatures",
                    result["spectral_signatures"]
                )
            ]

            for method, metrics in methods:

                f.write(
                    f"{method}\n"
                )

                f.write(
                    f"  Detection Accuracy: "
                    f"{metrics['Detection Accuracy']:.2f}%\n"
                )

                f.write(
                    f"  TPR: "
                    f"{metrics['TPR']:.2f}%\n"
                )

                f.write(
                    f"  FPR: "
                    f"{metrics['FPR']:.2f}%\n"
                )

                f.write(
                    f"  TP: {metrics['TP']}\n"
                )

                f.write(
                    f"  TN: {metrics['TN']}\n"
                )

                f.write(
                    f"  FP: {metrics['FP']}\n"
                )

                f.write(
                    f"  FN: {metrics['FN']}\n\n"
                )

    print(
        f"\nResults saved to:\n{output_file}"
    )


# ============================================================
# Final summary
# ============================================================

def print_final_summary(all_results):

    print("\n\n")
    print("=" * 95)
    print("FINAL INJECTION RATE VS DEFENCE EFFECTIVENESS")
    print("=" * 95)

    print(
        f"{'Rate':<10}"
        f"{'Method':<25}"
        f"{'Detection Acc.':<18}"
        f"{'TPR':<12}"
        f"{'FPR':<12}"
    )

    print("-" * 95)

    for result in all_results:

        rate = f"{result['rate'] * 100:.0f}%"

        methods = [
            (
                "Isolation Forest",
                result["isolation_forest"]
            ),
            (
                "Activation Clustering",
                result["activation_clustering"]
            ),
            (
                "Spectral Signatures",
                result["spectral_signatures"]
            )
        ]

        for method, metrics in methods:

            print(
                f"{rate:<10}"
                f"{method:<25}"
                f"{metrics['Detection Accuracy']:<18.2f}"
                f"{metrics['TPR']:<12.2f}"
                f"{metrics['FPR']:<12.2f}"
            )

        print("-" * 95)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("INJECTION RATE VS DEFENCE EFFECTIVENESS")
    print("=" * 70)

    print(
        f"\nUsing device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load dataset once
    # --------------------------------------------------------

    base_dataset = load_dataset()

    all_results = []

    # --------------------------------------------------------
    # Run all poisoning rates
    # --------------------------------------------------------

    for config in EXPERIMENTS:

        result = run_single_experiment(
            base_dataset,
            config
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        all_results
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print_final_summary(
        all_results
    )

    print("\nExperiment completed successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()