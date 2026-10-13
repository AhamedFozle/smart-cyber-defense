import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from attacks.badnets import add_trigger


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64

TARGET_CLASS = 0

TRIGGER_SIZE = 3

TRIGGER_VALUE = 1.0

MODEL_PATH = "models/full_filtered_cnn_best.pth"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# CIFAR-10 normalization values used during training
MEAN = (
    0.4914,
    0.4822,
    0.4465
)

STD = (
    0.2470,
    0.2435,
    0.2616
)


# ============================================================
# DATA TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )
])


# ============================================================
# LOAD CLEAN CIFAR-10 TEST DATA
# ============================================================

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


# ============================================================
# LOAD FULL FILTERED / RECOVERED MODEL
# ============================================================

print(f"Using device: {DEVICE}")


model = SimpleCNN(
    num_classes=10
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Full filtered/recovered CNN loaded successfully!"
)


# ============================================================
# EVALUATE CLEAN ACCURACY
# ============================================================

def evaluate_clean_accuracy():

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


    clean_accuracy = (
        100.0 * correct / total
    )


    return (
        clean_accuracy,
        correct,
        total
    )


# ============================================================
# ADD TRIGGER CORRECTLY TO NORMALIZED IMAGES
# ============================================================

def add_normalized_trigger(images):

    """
    Adds the BadNets trigger to images that have already
    been normalized using the CIFAR-10 mean and std values.

    The trigger is first defined as pixel value 1.0 in the
    original image space and then converted into normalized
    tensor space.
    """

    triggered_images = images.clone()


    # Convert original trigger value into normalized values
    normalized_trigger_values = []


    for mean, std in zip(MEAN, STD):

        normalized_value = (
            TRIGGER_VALUE - mean
        ) / std


        normalized_trigger_values.append(
            normalized_value
        )


    # Add trigger to bottom-right corner
    for channel in range(3):

        triggered_images[
            :,
            channel,
            -TRIGGER_SIZE:,
            -TRIGGER_SIZE:
        ] = normalized_trigger_values[channel]


    return triggered_images


# ============================================================
# EVALUATE ATTACK SUCCESS RATE
# ============================================================

def evaluate_attack_success_rate():

    model.eval()

    successful_attacks = 0

    total_attacked_samples = 0


    with torch.no_grad():

        for images, labels in test_loader:


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            # ------------------------------------------------
            # IMPORTANT:
            # Do not attack images already belonging to
            # the target class.
            # ------------------------------------------------

            non_target_mask = (
                labels != TARGET_CLASS
            )


            clean_images = images[
                non_target_mask
            ]


            if clean_images.size(0) == 0:

                continue


            # ------------------------------------------------
            # ADD BADNETS TRIGGER
            # ------------------------------------------------

            triggered_images = (
                add_normalized_trigger(
                    clean_images
                )
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            outputs = model(
                triggered_images
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            # ------------------------------------------------
            # COUNT SUCCESSFUL ATTACKS
            # ------------------------------------------------

            successful_attacks += (
                predicted == TARGET_CLASS
            ).sum().item()


            total_attacked_samples += (
                clean_images.size(0)
            )


    attack_success_rate = (
        100.0
        * successful_attacks
        / total_attacked_samples
    )


    return (
        attack_success_rate,
        successful_attacks,
        total_attacked_samples
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

print(
    "\nEvaluating Clean Accuracy..."
)


clean_accuracy, correct_clean, total_clean = (
    evaluate_clean_accuracy()
)


print(
    "\nEvaluating Attack Success Rate..."
)


attack_success_rate, successful_attacks, total_attacked = (
    evaluate_attack_success_rate()
)


# ============================================================
# GET ORIGINAL RESULTS
# ============================================================

original_clean_accuracy = 78.40

original_attack_success_rate = 95.77


# Calculate changes
clean_accuracy_change = (
    clean_accuracy
    - original_clean_accuracy
)


asr_reduction = (
    original_attack_success_rate
    - attack_success_rate
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "FULL FILTERED / RECOVERED MODEL RESULTS"
)

print("=" * 60)


print(
    f"Model: Full Filtered CNN"
)


print(
    f"Target Class: {TARGET_CLASS}"
)


print("-" * 60)


print(
    f"Clean Accuracy: "
    f"{clean_accuracy:.2f}%"
)


print(
    f"Correct Clean Predictions: "
    f"{correct_clean}/{total_clean}"
)


print("-" * 60)


print(
    f"Successful Attacks: "
    f"{successful_attacks}/{total_attacked}"
)


print(
    f"Attack Success Rate: "
    f"{attack_success_rate:.2f}%"
)


print("=" * 60)


# ============================================================
# RECOVERY COMPARISON
# ============================================================

print("\nRECOVERY COMPARISON")

print("=" * 60)


print(
    f"Original Backdoored CA: "
    f"{original_clean_accuracy:.2f}%"
)


print(
    f"Recovered Model CA:     "
    f"{clean_accuracy:.2f}%"
)


print(
    f"CA Change:              "
    f"{clean_accuracy_change:+.2f}%"
)


print("-" * 60)


print(
    f"Original Backdoored ASR: "
    f"{original_attack_success_rate:.2f}%"
)


print(
    f"Recovered Model ASR:     "
    f"{attack_success_rate:.2f}%"
)


print(
    f"ASR Reduction:           "
    f"{asr_reduction:.2f} percentage points"
)


print("=" * 60)