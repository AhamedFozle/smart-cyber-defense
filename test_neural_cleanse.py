import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from defenses.neural_cleanse import optimize_trigger


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64

NUM_SAMPLES = 512

ITERATIONS = 100

LEARNING_RATE = 0.1

MASK_WEIGHT = 0.01

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print(f"Using device: {DEVICE}")


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
# CIFAR-10
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# SELECT SAMPLES
# ============================================================

images = []

for i in range(NUM_SAMPLES):

    image, _ = dataset[i]

    images.append(image)


images = torch.stack(
    images
).to(DEVICE)


print(
    f"\nUsing {NUM_SAMPLES} clean test images."
)


# ============================================================
# NEURAL CLEANSE
# ============================================================

results = {}


print("\n" + "=" * 60)

print("NEURAL CLEANSE ANALYSIS")

print("=" * 60)


for target_class in range(10):

    print(
        f"\nOptimizing trigger for "
        f"target class {target_class}..."
    )

    result = optimize_trigger(
    model=model,

    images=images,

    target_class=target_class,

    device=DEVICE,

    patch_size=3,

    iterations=ITERATIONS,

    learning_rate=LEARNING_RATE
)

    results[target_class] = result

    print(
        f"Target Class: {target_class}"
    )

    print(
        f"Mask Norm: "
        f"{result['mask_norm']:.6f}"
    )

    print(
        f"Attack Success Rate: "
        f"{result['attack_success_rate']:.2f}%"
    )


# ============================================================
# ANALYZE MASK NORMS
# ============================================================

mask_norms = torch.tensor([
    results[c]["mask_norm"]
    for c in range(10)
])


median_norm = torch.median(
    mask_norms
).item()


# Smaller mask = more suspicious
anomaly_scores = (
    median_norm /
    (mask_norms + 1e-8)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("NEURAL CLEANSE RESULTS")

print("=" * 60)


for target_class in range(10):

    print(
        f"Class {target_class}: "
        f"Mask Norm = "
        f"{results[target_class]['mask_norm']:.6f}, "
        f"ASR = "
        f"{results[target_class]['attack_success_rate']:.2f}%, "
        f"Anomaly Score = "
        f"{anomaly_scores[target_class]:.4f}"
    )


# ============================================================
# MOST SUSPICIOUS CLASS
# ============================================================

suspicious_class = torch.argmax(
    anomaly_scores
).item()


print("\n" + "=" * 60)

print("NEURAL CLEANSE DETECTION")

print("=" * 60)

print(
    f"Most suspicious target class: "
    f"{suspicious_class}"
)

print(
    f"Highest anomaly score: "
    f"{anomaly_scores[suspicious_class]:.4f}"
)

print(
    f"Expected BadNets target class: "
    f"0"
)

print("=" * 60)


if suspicious_class == 0:

    print(
        "\nRESULT: Neural Cleanse identified "
        "the expected target class 0."
    )

else:

    print(
        "\nRESULT: Neural Cleanse did not identify "
        "class 0 as the most suspicious class."
    )