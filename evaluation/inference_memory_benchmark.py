import os
import sys
import time
import csv
import gc

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ============================================================
# IMPORT MODEL
# ============================================================

from models.cnn import SimpleCNN


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 64

NUM_WARMUP_BATCHES = 5
NUM_TIMED_BATCHES = 50

IMAGE_SIZE = 32

# ============================================================
# MODEL CHECKPOINTS
# ============================================================

MODELS = {
    "Baseline": os.path.join(
        MODELS_DIR,
        "backdoored_cnn_10percent_best.pth"
    ),

    "Fine-Pruning 20%": os.path.join(
        MODELS_DIR,
        "fine_pruned_cnn_20percent_best.pth"
    ),

    "Fine-Pruning 30%": os.path.join(
        MODELS_DIR,
        "fine_pruned_cnn_30percent_best.pth"
    ),

    "Fine-Pruning 40%": os.path.join(
        MODELS_DIR,
        "fine_pruned_cnn_40percent_best.pth"
    ),
}


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

print("\nLoading CIFAR-10 test dataset...")

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {len(test_dataset)}")


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_model(checkpoint_path):

    model = SimpleCNN(num_classes=10)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# MODEL SIZE
# ============================================================

def calculate_model_size(model):

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    parameter_memory_mb = (
        total_parameters * 4
    ) / (1024 ** 2)

    return total_parameters, parameter_memory_mb


# ============================================================
# PROCESS MEMORY
# ============================================================

def get_process_memory_mb():

    process = psutil.Process(
        os.getpid()
    )

    memory_bytes = process.memory_info().rss

    return memory_bytes / (1024 ** 2)


# ============================================================
# INFERENCE LATENCY
# ============================================================

def measure_inference_latency(model):

    model.eval()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\nPerforming warm-up inference...")

    warmup_count = 0

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(DEVICE)

            _ = model(images)

            warmup_count += 1

            if warmup_count >= NUM_WARMUP_BATCHES:
                break

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    batch_times = []

    print("Measuring inference latency...")

    with torch.no_grad():

        for batch_number, (images, _) in enumerate(
            test_loader
        ):

            if batch_number >= NUM_TIMED_BATCHES:
                break

            images = images.to(DEVICE)

            start_time = time.perf_counter()

            _ = model(images)

            end_time = time.perf_counter()

            elapsed = end_time - start_time

            batch_times.append(elapsed)

    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------

    average_batch_time = (
        sum(batch_times) /
        len(batch_times)
    )

    average_image_time = (
        average_batch_time /
        BATCH_SIZE
    )

    latency_ms_per_batch = (
        average_batch_time * 1000
    )

    latency_ms_per_image = (
        average_image_time * 1000
    )

    return (
        latency_ms_per_batch,
        latency_ms_per_image
    )


# ============================================================
# MEMORY MEASUREMENT
# ============================================================

def measure_memory(model):

    gc.collect()

    before_memory = get_process_memory_mb()

    # Run inference to ensure model/data are active
    with torch.no_grad():

        for batch_number, (images, _) in enumerate(
            test_loader
        ):

            images = images.to(DEVICE)

            _ = model(images)

            if batch_number >= 10:
                break

    after_memory = get_process_memory_mb()

    memory_increase = (
        after_memory -
        before_memory
    )

    return (
        before_memory,
        after_memory,
        memory_increase
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("INFERENCE LATENCY AND MEMORY BENCHMARK")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Timed batches: {NUM_TIMED_BATCHES}"
    )

    results = []

    for model_name, checkpoint_path in MODELS.items():

        print("\n")
        print("-" * 70)
        print(f"Testing: {model_name}")
        print("-" * 70)

        if not os.path.exists(checkpoint_path):

            print(
                f"WARNING: Checkpoint not found:"
            )
            print(checkpoint_path)

            continue

        print("Loading model...")

        model = load_model(
            checkpoint_path
        )

        total_parameters, parameter_memory_mb = (
            calculate_model_size(model)
        )

        print(
            f"Parameters: {total_parameters:,}"
        )

        print(
            f"Parameter memory: "
            f"{parameter_memory_mb:.2f} MB"
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        print("\nMeasuring memory usage...")

        (
            memory_before,
            memory_after,
            memory_increase
        ) = measure_memory(model)

        print(
            f"Memory before inference: "
            f"{memory_before:.2f} MB"
        )

        print(
            f"Memory after inference: "
            f"{memory_after:.2f} MB"
        )

        print(
            f"Memory increase: "
            f"{memory_increase:.2f} MB"
        )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        (
            latency_batch,
            latency_image
        ) = measure_inference_latency(model)

        print(
            f"\nAverage batch latency: "
            f"{latency_batch:.4f} ms"
        )

        print(
            f"Average image latency: "
            f"{latency_image:.4f} ms"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "Parameters": total_parameters,
            "Parameter_Memory_MB": parameter_memory_mb,
            "Memory_Before_MB": memory_before,
            "Memory_After_MB": memory_after,
            "Memory_Increase_MB": memory_increase,
            "Batch_Latency_MS": latency_batch,
            "Image_Latency_MS": latency_image
        })

        # Free memory
        del model

        gc.collect()

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        RESULTS_DIR,
        "inference_memory_benchmark.csv"
    )

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(results)

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    txt_path = os.path.join(
        RESULTS_DIR,
        "inference_memory_benchmark.txt"
    )

    with open(
        txt_path,
        "w"
    ) as file:

        file.write(
            "Inference Latency and Memory Benchmark\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Batch Size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Timed Batches: {NUM_TIMED_BATCHES}\n\n"
        )

        for result in results:

            file.write(
                f"Model: {result['Model']}\n"
            )

            file.write(
                f"Parameters: "
                f"{result['Parameters']:,}\n"
            )

            file.write(
                f"Parameter Memory: "
                f"{result['Parameter_Memory_MB']:.2f} MB\n"
            )

            file.write(
                f"Memory Before: "
                f"{result['Memory_Before_MB']:.2f} MB\n"
            )

            file.write(
                f"Memory After: "
                f"{result['Memory_After_MB']:.2f} MB\n"
            )

            file.write(
                f"Memory Increase: "
                f"{result['Memory_Increase_MB']:.2f} MB\n"
            )

            file.write(
                f"Batch Latency: "
                f"{result['Batch_Latency_MS']:.4f} ms\n"
            )

            file.write(
                f"Image Latency: "
                f"{result['Image_Latency_MS']:.4f} ms\n"
            )

            file.write("\n")

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)

    print("\nResults:")

    for result in results:

        print(
            f"\n{result['Model']}"
        )

        print(
            f"  Image latency: "
            f"{result['Image_Latency_MS']:.4f} ms"
        )

        print(
            f"  Memory increase: "
            f"{result['Memory_Increase_MB']:.2f} MB"
        )

    print("\nSaved files:")

    print(csv_path)
    print(txt_path)


if __name__ == "__main__":
    main()