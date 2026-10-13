import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from defenses.abs_style import run_abs_style


# ==========================================================
# CONFIGURATION
# ==========================================================

BATCH_SIZE = 64
NUM_SAMPLES = 64

MODEL_PATH = "models/backdoored_cnn_best.pth"

TARGET_CLASS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("ABS-STYLE MODEL-LEVEL DEFENSE")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    # ------------------------------------------------------
    # CIFAR-10 normalization
    # ------------------------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    # ------------------------------------------------------
    # Load clean CIFAR-10 test dataset
    # ------------------------------------------------------

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Use a small clean subset for CPU efficiency
    test_subset = torch.utils.data.Subset(
        test_dataset,
        range(NUM_SAMPLES)
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    images, labels = next(iter(test_loader))

    print(f"\nImages used for ABS analysis: {len(images)}")

    # ------------------------------------------------------
    # Load backdoored CNN
    # ------------------------------------------------------

    model = SimpleCNN(num_classes=10)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    print("Backdoored CNN loaded successfully!")

    # ------------------------------------------------------
    # Run ABS-style analysis
    # ------------------------------------------------------

    top_results, all_results = run_abs_style(
        model=model,
        images=images,
        device=DEVICE,
        stimulation_factor=2.0,
        top_k=10
    )

    # ------------------------------------------------------
    # Check whether target class appears in suspicious list
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("ABS-STYLE DETECTION SUMMARY")
    print("=" * 60)

    target_results = [
        result
        for result in top_results
        if result["associated_class"] == TARGET_CLASS
    ]

    if target_results:

        print(
            f"Target class {TARGET_CLASS} detected "
            f"among the top suspicious channels."
        )

        print("\nSuspicious channels associated with target class:")

        for result in target_results:

            print(
                f"Layer: {result['layer']} | "
                f"Channel: {result['channel']} | "
                f"Score: {result['logit_change']:.4f}"
            )

    else:

        print(
            f"Target class {TARGET_CLASS} was NOT found "
            f"among the top suspicious channels."
        )

    # ------------------------------------------------------
    # Highest suspicious neuron
    # ------------------------------------------------------

    if len(top_results) > 0:

        highest = top_results[0]

        print("\nMost suspicious channel:")
        print(f"Layer: {highest['layer']}")
        print(f"Channel: {highest['channel']}")
        print(f"Associated class: {highest['associated_class']}")
        print(f"Stimulation score: {highest['logit_change']:.4f}")

    print("\nABS-style analysis completed!")


if __name__ == "__main__":
    main()