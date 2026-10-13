import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import add_trigger


# ============================================
# Configuration
# ============================================

TARGET_CLASS = 0
TRIGGER_SIZE = 3
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================
# Load filtered model
# ============================================

print(f"Using device: {DEVICE}")

model = SimpleCNN(num_classes=10).to(DEVICE)

checkpoint = torch.load(
    "models/filtered_cnn_best.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Filtered/retrained CNN loaded successfully!")
print(
    f"Model Clean Accuracy: "
    f"{checkpoint['clean_accuracy']:.2f}%"
)


# ============================================
# CIFAR-10 test dataset
# ============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# Evaluate ASR
# ============================================

successful_attacks = 0
total_attack_samples = 0

print("\nEvaluating Attack Success Rate...")


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Exclude samples already belonging
        # to the target class
        mask = labels != TARGET_CLASS

        images = images[mask]

        if images.size(0) == 0:
            continue

        # Add BadNets trigger
        triggered_images = add_trigger(
            images,
            trigger_size=TRIGGER_SIZE
        )

        # Model prediction
        outputs = model(triggered_images)

        _, predicted = torch.max(
            outputs,
            1
        )

        successful_attacks += (
            predicted == TARGET_CLASS
        ).sum().item()

        total_attack_samples += (
            predicted.size(0)
        )


# ============================================
# Calculate ASR
# ============================================

asr = (
    successful_attacks /
    total_attack_samples
) * 100


# ============================================
# Final results
# ============================================

print("\n" + "=" * 60)
print("FILTERED / RECOVERED MODEL ASR RESULTS")
print("=" * 60)

print(
    f"Clean Accuracy: "
    f"{checkpoint['clean_accuracy']:.2f}%"
)

print(f"Target Class: {TARGET_CLASS}")

print(
    f"Successful Attacks: "
    f"{successful_attacks}/{total_attack_samples}"
)

print(f"Attack Success Rate: {asr:.2f}%")

print("=" * 60)