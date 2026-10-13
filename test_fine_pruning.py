
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from recovery.fine_pruning import FinePruner


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64

# Fine-tuning settings
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.0005

# Fine-pruning rates to test
PRUNING_RATES = [0.20, 0.30, 0.40]

# BadNets configuration
POISON_RATE = 0.10
TARGET_CLASS = 0
TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0

# Use a subset for faster activation collection and fine-tuning.
# Evaluation still uses the complete CIFAR-10 test set.
CLEAN_SUBSET_SIZE = 10000

# Number of workers for Windows/CPU
NUM_WORKERS = 0

# Paths
BACKDOORED_MODEL_PATH = "models/backdoored_cnn_best.pth"
RESULTS_DIR = "results"
MODEL_DIR = "models"


# CIFAR-10 normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Makes CPU/GPU behavior more reproducible where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA TRANSFORMS
# ============================================================

# Training transform for clean fine-tuning.
# We use light augmentation to improve generalization.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])


# Deterministic transform for evaluation and activation collection.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])


# ============================================================
# BADNETS TRIGGER
# ============================================================

def add_trigger(image, trigger_size=3, trigger_value=1.0):
    """
    Add a square BadNets trigger to the bottom-right corner.

    image:
        Tensor with shape [C, H, W]

    Returns:
        Triggered image tensor.
    """

    image = image.clone()

    _, height, width = image.shape

    image[
        :,
        height - trigger_size:height,
        width - trigger_size:width
    ] = trigger_value

    return image


# ============================================================
# CREATE BADNETS TEST DATA
# ============================================================

class BadNetsTestDataset(torch.utils.data.Dataset):
    """
    Creates a triggered test dataset.

    Only non-target-class images are included because standard ASR
    evaluation measures how often a trigger causes originally
    non-target images to be classified as the attacker's target class.
    """

    def __init__(
        self,
        dataset,
        target_class=0,
        trigger_size=3,
        trigger_value=1.0
    ):
        self.dataset = dataset
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        # Keep only non-target samples
        self.indices = [
            i
            for i in range(len(dataset))
            if dataset.targets[i] != target_class
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_index = self.indices[idx]

        image, original_label = self.dataset[original_index]

        triggered_image = add_trigger(
            image,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value
        )

        # The expected attack target is TARGET_CLASS
        return triggered_image, self.target_class


# ============================================================
# MODEL LOADING
# ============================================================

def load_backdoored_model():
    """
    Load the previously trained BadNets model.
    """

    model = SimpleCNN(num_classes=10)

    checkpoint = torch.load(
        BACKDOORED_MODEL_PATH,
        map_location=device
    )

    # Support both:
    # 1. state_dict directly
    # 2. checkpoint dictionary containing state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    return model


# ============================================================
# CLEAN ACCURACY
# ============================================================

def evaluate_clean_accuracy(model, loader):
    """
    Evaluate Clean Accuracy (CA).
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# ATTACK SUCCESS RATE
# ============================================================

def evaluate_asr(model, loader, target_class=0):
    """
    Calculate Attack Success Rate.

    Only non-target clean test images are used.

    ASR =
        number of triggered non-target images classified as target
        ---------------------------------------------------------
        total number of triggered non-target images
    """

    model.eval()

    successful_attacks = 0
    total = 0

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            successful_attacks += (
                predictions == target_class
            ).sum().item()

            total += images.size(0)

    asr = 100.0 * successful_attacks / total

    return asr, successful_attacks, total


# ============================================================
# BUILD DATA LOADERS
# ============================================================

def create_dataloaders():
    """
    Create:
      - clean training dataset
      - clean test dataset
      - BadNets triggered test dataset
    """

    print("\nLoading CIFAR-10 datasets...")

    clean_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # Separate deterministic dataset for activation collection
    clean_activation_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=test_transform
    )

    clean_test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # --------------------------------------------------------
    # Fixed clean subset
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    subset_size = min(
        CLEAN_SUBSET_SIZE,
        len(clean_train_dataset)
    )

    subset_indices = torch.randperm(
        len(clean_train_dataset),
        generator=generator
    )[:subset_size].tolist()

    fine_tune_dataset = Subset(
        clean_train_dataset,
        subset_indices
    )

    activation_dataset = Subset(
        clean_activation_dataset,
        subset_indices
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    fine_tune_loader = DataLoader(
        fine_tune_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    activation_loader = DataLoader(
        activation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # BadNets test loader
    # --------------------------------------------------------

    badnets_test_dataset = BadNetsTestDataset(
        clean_test_dataset,
        target_class=TARGET_CLASS,
        trigger_size=TRIGGER_SIZE,
        trigger_value=TRIGGER_VALUE
    )

    badnets_test_loader = DataLoader(
        badnets_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Clean training dataset: {len(clean_train_dataset)} samples"
    )

    print(
        f"Fine-tuning subset: {len(fine_tune_dataset)} samples"
    )

    print(
        f"Activation collection subset: {len(activation_dataset)} samples"
    )

    print(
        f"Clean test dataset: {len(clean_test_dataset)} samples"
    )

    print(
        f"BadNets ASR test dataset: {len(badnets_test_dataset)} samples"
    )

    return (
        fine_tune_loader,
        activation_loader,
        clean_test_loader,
        badnets_test_loader
    )


# ============================================================
# FINE-TUNING
# ============================================================

def fine_tune_model(
    model,
    train_loader,
    clean_test_loader,
    masks,
    epochs=10,
    learning_rate=0.0005
):
    """
    Fine-tune the pruned model using CLEAN data.

    After every optimizer update, the pruning masks are reapplied
    so that pruned filters remain permanently disabled.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_ca = -1.0
    best_state = None
    best_epoch = 0

    print("\nFINE-TUNING ON CLEAN DATA")

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        start_epoch = time.time()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # IMPORTANT:
            # Keep pruned filters permanently zero.
            pruner.reapply_masks(masks)
            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        clean_accuracy = evaluate_clean_accuracy(
            model,
            clean_test_loader
        )

        epoch_time = time.time() - start_epoch

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Clean CA: {clean_accuracy:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model based on clean accuracy
        if clean_accuracy > best_ca:

            best_ca = clean_accuracy
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print(
                f"  -> New best recovered model! "
                f"CA: {best_ca:.2f}%"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Make sure masks are still applied
    pruner.reapply_masks(masks)

    print(
        f"\nBest fine-tuned epoch: {best_epoch}"
    )

    print(
        f"Best Clean Accuracy: {best_ca:.2f}%"
    )

    return model, best_ca, best_epoch


# ============================================================
# COUNT PRUNED FILTERS
# ============================================================

def calculate_pruning_statistics(masks):
    """
    Calculate actual pruning statistics from masks.
    """

    total_channels = 0
    pruned_channels = 0

    layer_statistics = []

    for layer_name, mask in masks.items():

        total = mask.numel()
        remaining = mask.sum().item()
        pruned = total - remaining

        total_channels += total
        pruned_channels += pruned

        layer_statistics.append(
            (
                layer_name,
                int(pruned),
                int(total)
            )
        )

    actual_rate = (
        100.0 * pruned_channels / total_channels
        if total_channels > 0
        else 0.0
    )

    return (
        pruned_channels,
        total_channels,
        actual_rate,
        layer_statistics
    )


# ============================================================
# VERIFY MASKS
# ============================================================

def verify_pruned_weights(model, masks):
    """
    Verify that all pruned filters remain zero.
    """

    max_pruned_weight = 0.0

    for layer_name, mask in masks.items():

        layer = getattr(model, layer_name)

        weight = layer.weight.detach()

        # mask shape:
        # [out_channels]
        #
        # Convert to [out_channels, 1, 1, 1]
        expanded_mask = mask.to(
            weight.device
        ).view(-1, 1, 1, 1)

        pruned_weights = (
            weight * (1.0 - expanded_mask)
        )

        if pruned_weights.numel() > 0:

            max_value = (
                pruned_weights.abs().max().item()
            )

            max_pruned_weight = max(
                max_pruned_weight,
                max_value
            )

    return max_pruned_weight


# ============================================================
# SAVE RESULTS
# ============================================================

def save_result(
    pruning_rate,
    actual_pruning_rate,
    baseline_ca,
    recovered_ca,
    baseline_asr,
    recovered_asr,
    asr_reduction,
    ca_change,
    recovery_rate,
    pruning_time,
    fine_tuning_time,
    total_time,
    best_epoch
):
    """
    Save one pruning experiment result to a text file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    filename = (
        f"{RESULTS_DIR}/"
        f"fine_pruning_{int(pruning_rate * 100)}percent.txt"
    )

    with open(filename, "w", encoding="utf-8") as file:

        file.write("FINE-PRUNING RECOVERY RESULT\n")
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Configured pruning rate: "
            f"{pruning_rate * 100:.2f}%\n"
        )

        file.write(
            f"Actual pruning rate: "
            f"{actual_pruning_rate:.2f}%\n\n"
        )

        file.write(
            f"Baseline Clean Accuracy: "
            f"{baseline_ca:.2f}%\n"
        )

        file.write(
            f"Recovered Clean Accuracy: "
            f"{recovered_ca:.2f}%\n"
        )

        file.write(
            f"CA Change: "
            f"{ca_change:+.2f} percentage points\n\n"
        )

        file.write(
            f"Baseline ASR: "
            f"{baseline_asr:.2f}%\n"
        )

        file.write(
            f"Recovered ASR: "
            f"{recovered_asr:.2f}%\n"
        )

        file.write(
            f"ASR Reduction: "
            f"{asr_reduction:+.2f} percentage points\n"
        )

        file.write(
            f"Recovery Rate: "
            f"{recovery_rate:+.2f}%\n\n"
        )

        file.write(
            f"Best Fine-tuning Epoch: "
            f"{best_epoch}\n\n"
        )

        file.write(
            f"Pruning Time: "
            f"{pruning_time:.2f} seconds\n"
        )

        file.write(
            f"Fine-tuning Time: "
            f"{fine_tuning_time:.2f} seconds\n"
        )

        file.write(
            f"Total Experiment Time: "
            f"{total_time:.2f} seconds\n"
        )

    print(
        f"Result saved to: {filename}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    global pruner

    set_seed(SEED)

    experiment_start = time.time()

    print("=" * 70)
    print("FINE-PRUNING MODEL RECOVERY")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Random seed: {SEED}")

    print("\nConfiguration:")
    print(f"  BadNets poison rate: {POISON_RATE * 100:.1f}%")
    print(f"  Target class: {TARGET_CLASS}")
    print(f"  Trigger size: {TRIGGER_SIZE}x{TRIGGER_SIZE}")
    print(f"  Trigger value: {TRIGGER_VALUE}")
    print(f"  Fine-tuning epochs: {FINE_TUNE_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Clean subset: {CLEAN_SUBSET_SIZE}")
    print(
        f"  Pruning rates to test: "
        f"{[int(x * 100) for x in PRUNING_RATES]}%"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(BACKDOORED_MODEL_PATH):

        print(
            f"\nERROR: Backdoored model not found:"
        )

        print(
            f"  {BACKDOORED_MODEL_PATH}"
        )

        print(
            "\nPlease run train_badnets.py first."
        )

        return

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    (
        fine_tune_loader,
        activation_loader,
        clean_test_loader,
        badnets_test_loader
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Load original backdoored model
    # --------------------------------------------------------

    print("\nLoading backdoored model...")

    original_model = load_backdoored_model()

    print(
        f"Loaded: {BACKDOORED_MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Baseline evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE BACKDOORED MODEL")
    print("=" * 70)

    baseline_ca = evaluate_clean_accuracy(
        original_model,
        clean_test_loader
    )

    baseline_asr, baseline_success, baseline_total = (
        evaluate_asr(
            original_model,
            badnets_test_loader,
            target_class=TARGET_CLASS
        )
    )

    print(
        f"Baseline Clean Accuracy (CA): "
        f"{baseline_ca:.2f}%"
    )

    print(
        f"Baseline ASR: "
        f"{baseline_asr:.2f}%"
    )

    print(
        f"Successful attacks: "
        f"{baseline_success}/{baseline_total}"
    )

    # --------------------------------------------------------
    # Create FinePruner
    # --------------------------------------------------------

    print("\nInitializing FinePruner...")

    pruner = FinePruner(original_model,device)

    # --------------------------------------------------------
    # Run each pruning rate
    # --------------------------------------------------------

    all_results = []

    for configured_rate in PRUNING_RATES:

        print("\n\n")
        print("=" * 70)
        print(
            f"FINE-PRUNING EXPERIMENT: "
            f"{int(configured_rate * 100)}%"
        )
        print("=" * 70)

        set_seed(SEED)

        experiment_start_rate = time.time()

        # ----------------------------------------------------
        # Reload fresh backdoored model
        # ----------------------------------------------------

        model = load_backdoored_model()

        # Create a fresh pruner for this model
        pruner = FinePruner(model,device)

        # ----------------------------------------------------
        # Collect activations on CLEAN data
        # ----------------------------------------------------

        print("\nCollecting clean-data activations...")

        activation_start = time.time()

        masks = pruner.prune(
            activation_loader,
            configured_rate
        )

        pruning_time = time.time() - activation_start

        # ----------------------------------------------------
        # Pruning statistics
        # ----------------------------------------------------

        (
            pruned_channels,
            total_channels,
            actual_pruning_rate,
            layer_statistics
        ) = calculate_pruning_statistics(masks)

        print("\nPRUNING SUMMARY")

        for layer_name, pruned, total in layer_statistics:

            print(
                f"  {layer_name}: "
                f"{pruned}/{total} filters pruned"
            )

        print(
            f"\nTotal pruned channels: "
            f"{pruned_channels}/{total_channels}"
        )

        print(
            f"Configured pruning rate: "
            f"{configured_rate * 100:.2f}%"
        )

        print(
            f"Actual pruning rate: "
            f"{actual_pruning_rate:.2f}%"
        )

        # ----------------------------------------------------
        # Verify masks immediately
        # ----------------------------------------------------

        max_pruned_weight = verify_pruned_weights(
            model,
            masks
        )

        print(
            f"Maximum pruned-filter weight: "
            f"{max_pruned_weight}"
        )

        # ----------------------------------------------------
        # Immediate post-pruning evaluation
        # ----------------------------------------------------

        print("\nIMMEDIATELY AFTER PRUNING")

        immediate_ca = evaluate_clean_accuracy(
            model,
            clean_test_loader
        )

        immediate_asr, _, _ = evaluate_asr(
            model,
            badnets_test_loader,
            target_class=TARGET_CLASS
        )

        print(
            f"Clean Accuracy: "
            f"{immediate_ca:.2f}%"
        )

        print(
            f"ASR: "
            f"{immediate_asr:.2f}%"
        )

        # ----------------------------------------------------
        # Fine-tune on CLEAN data
        # ----------------------------------------------------

        fine_tune_start = time.time()

        (
            model,
            best_ca,
            best_epoch
        ) = fine_tune_model(
            model=model,
            train_loader=fine_tune_loader,
            clean_test_loader=clean_test_loader,
            masks=masks,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=LEARNING_RATE
        )

        fine_tuning_time = time.time() - fine_tune_start

        # ----------------------------------------------------
        # Final mask verification
        # ----------------------------------------------------

        max_pruned_weight = verify_pruned_weights(
            model,
            masks
        )

        print(
            "\nFinal maximum pruned-filter weight: "
            f"{max_pruned_weight}"
        )

        # ----------------------------------------------------
        # Final evaluation
        # ----------------------------------------------------

        print("\nFINAL RECOVERED MODEL EVALUATION")

        recovered_ca = evaluate_clean_accuracy(
            model,
            clean_test_loader
        )

        recovered_asr, recovered_success, recovered_total = (
            evaluate_asr(
                model,
                badnets_test_loader,
                target_class=TARGET_CLASS
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        asr_reduction = (
            baseline_asr - recovered_asr
        )

        ca_change = (
            recovered_ca - baseline_ca
        )

        # Recovery rate:
        # Positive = backdoor ASR decreased
        # Negative = backdoor ASR increased
        if baseline_asr > 0:

            recovery_rate = (
                (baseline_asr - recovered_asr)
                / baseline_asr
            ) * 100.0

        else:

            recovery_rate = 0.0

        total_time = time.time() - experiment_start_rate

        # ----------------------------------------------------
        # Print final results
        # ----------------------------------------------------

        print("\n" + "-" * 70)
        print(
            f"RESULTS: {int(configured_rate * 100)}% PRUNING"
        )
        print("-" * 70)

        print(
            f"Baseline CA:       {baseline_ca:.2f}%"
        )

        print(
            f"Recovered CA:      {recovered_ca:.2f}%"
        )

        print(
            f"CA Change:         {ca_change:+.2f} pp"
        )

        print()

        print(
            f"Baseline ASR:      {baseline_asr:.2f}%"
        )

        print(
            f"Recovered ASR:     {recovered_asr:.2f}%"
        )

        print(
            f"ASR Reduction:     {asr_reduction:+.2f} pp"
        )

        print(
            f"Recovery Rate:     {recovery_rate:+.2f}%"
        )

        print()

        print(
            f"Successful attacks:"
            f" {recovered_success}/{recovered_total}"
        )

        print(
            f"Configured pruning:"
            f" {configured_rate * 100:.2f}%"
        )

        print(
            f"Actual pruning:"
            f" {actual_pruning_rate:.2f}%"
        )

        print(
            f"Best epoch:        {best_epoch}"
        )

        print(
            f"Pruning time:      {pruning_time:.2f}s"
        )

        print(
            f"Fine-tuning time:  {fine_tuning_time:.2f}s"
        )

        print(
            f"Total time:        {total_time:.2f}s"
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = (
            f"{MODEL_DIR}/"
            f"fine_pruned_cnn_{int(configured_rate * 100)}percent_best.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"\nRecovered model saved to:"
        )

        print(
            f"  {model_path}"
        )

        # ----------------------------------------------------
        # Save result text file
        # ----------------------------------------------------

        save_result(
            pruning_rate=configured_rate,
            actual_pruning_rate=actual_pruning_rate,
            baseline_ca=baseline_ca,
            recovered_ca=recovered_ca,
            baseline_asr=baseline_asr,
            recovered_asr=recovered_asr,
            asr_reduction=asr_reduction,
            ca_change=ca_change,
            recovery_rate=recovery_rate,
            pruning_time=pruning_time,
            fine_tuning_time=fine_tuning_time,
            total_time=total_time,
            best_epoch=best_epoch
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        all_results.append({
            "pruning_rate": configured_rate * 100,
            "actual_pruning_rate": actual_pruning_rate,
            "baseline_ca": baseline_ca,
            "recovered_ca": recovered_ca,
            "ca_change": ca_change,
            "baseline_asr": baseline_asr,
            "recovered_asr": recovered_asr,
            "asr_reduction": asr_reduction,
            "recovery_rate": recovery_rate,
            "best_epoch": best_epoch,
            "pruning_time": pruning_time,
            "fine_tuning_time": fine_tuning_time,
            "total_time": total_time
        })

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n\n")
    print("=" * 100)
    print("FINE-PRUNING COMPARISON")
    print("=" * 100)

    header = (
        f"{'Pruning':<10}"
        f"{'Actual':<10}"
        f"{'CA':<10}"
        f"{'ASR':<10}"
        f"{'ASR Red.':<12}"
        f"{'CA Change':<12}"
        f"{'Recovery':<12}"
    )

    print(header)
    print("-" * 100)

    for result in all_results:

        print(
            f"{result['pruning_rate']:.0f}%"
            f"{'':<6}"
            f"{result['actual_pruning_rate']:.2f}%"
            f"{'':<4}"
            f"{result['recovered_ca']:.2f}%"
            f"{'':<4}"
            f"{result['recovered_asr']:.2f}%"
            f"{'':<4}"
            f"{result['asr_reduction']:+.2f} pp"
            f"{'':<3}"
            f"{result['ca_change']:+.2f} pp"
            f"{'':<3}"
            f"{result['recovery_rate']:+.2f}%"
        )

    print("-" * 100)

    # ========================================================
    # BEST RECOVERY
    # ========================================================

    if all_results:

        # Highest recovery rate
        best_recovery = max(
            all_results,
            key=lambda x: x["recovery_rate"]
        )

        print("\nBEST BACKDOOR RECOVERY")

        print(
            f"Pruning rate: "
            f"{best_recovery['pruning_rate']:.0f}%"
        )

        print(
            f"Actual pruning: "
            f"{best_recovery['actual_pruning_rate']:.2f}%"
        )

        print(
            f"Recovered CA: "
            f"{best_recovery['recovered_ca']:.2f}%"
        )

        print(
            f"Recovered ASR: "
            f"{best_recovery['recovered_asr']:.2f}%"
        )

        print(
            f"ASR reduction: "
            f"{best_recovery['asr_reduction']:+.2f} pp"
        )

        print(
            f"Recovery rate: "
            f"{best_recovery['recovery_rate']:+.2f}%"
        )

    # ========================================================
    # FINAL EXPERIMENT SUMMARY
    # ========================================================

    total_experiment_time = (
        time.time() - experiment_start
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"Total experiment time: "
        f"{total_experiment_time:.2f} seconds"
    )

    print(
        f"Results directory: "
        f"{RESULTS_DIR}/"
    )

    print(
        f"Models directory: "
        f"{MODEL_DIR}/"
    )

    print("\nGenerated files include:")

    for result in all_results:

        rate = int(result["pruning_rate"])

        print(
            f"  models/fine_pruned_cnn_{rate}percent_best.pth"
        )

        print(
            f"  results/fine_pruning_{rate}percent.txt"
        )

    print("\nDone.")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()

