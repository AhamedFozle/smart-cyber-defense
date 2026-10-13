import numpy as np
import torch
import torch.nn.functional as F


def calculate_entropy(probabilities):
    """
    Calculate Shannon entropy for a probability distribution.
    """

    probabilities = np.asarray(probabilities)

    probabilities = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


def calculate_strip_entropy(
    model,
    image,
    reference_images,
    device,
    num_perturbations=10
):
    """
    Calculate STRIP entropy for one image.

    The image is intentionally mixed with reference images.
    """

    model.eval()

    entropies = []

    num_references = min(
        num_perturbations,
        len(reference_images)
    )

    with torch.no_grad():

        for i in range(num_references):

            reference = reference_images[i]

            # Normalize mixing operation
            perturbed = (
                0.5 * image
                + 0.5 * reference
            )

            perturbed = perturbed.unsqueeze(0).to(device)

            output = model(perturbed)

            probabilities = F.softmax(
                output,
                dim=1
            )

            probabilities = (
                probabilities
                .cpu()
                .numpy()[0]
            )

            entropy = calculate_entropy(
                probabilities
            )

            entropies.append(entropy)

    return float(
        np.mean(entropies)
    )


def calculate_strip_entropies(
    model,
    images,
    reference_images,
    device,
    num_perturbations=10
):
    """
    Calculate STRIP entropy for multiple images.
    """

    entropies = []

    total = len(images)

    print(
        f"\nCalculating STRIP entropy "
        f"for {total} samples..."
    )

    for i, image in enumerate(images):

        entropy = calculate_strip_entropy(
            model=model,
            image=image,
            reference_images=reference_images,
            device=device,
            num_perturbations=num_perturbations
        )

        entropies.append(entropy)

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{total}"
            )

    return np.array(
        entropies,
        dtype=np.float32
    )


def find_threshold(
    clean_entropies,
    poisoned_entropies
):
    """
    Find a threshold that maximizes
    clean/poisoned classification accuracy.

    Lower entropy is considered suspicious.
    """

    all_entropies = np.concatenate(
        [
            clean_entropies,
            poisoned_entropies
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(len(clean_entropies)),
            np.ones(len(poisoned_entropies))
        ]
    )

    best_threshold = None
    best_accuracy = -1

    for threshold in np.linspace(
        all_entropies.min(),
        all_entropies.max(),
        200
    ):

        predictions = (
            all_entropies < threshold
        ).astype(int)

        accuracy = np.mean(
            predictions == labels
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_accuracy * 100)
    )


def calculate_metrics(
    clean_entropies,
    poisoned_entropies,
    threshold
):
    """
    Calculate STRIP detection metrics.
    """

    clean_predictions = (
        clean_entropies < threshold
    ).astype(int)

    poisoned_predictions = (
        poisoned_entropies < threshold
    ).astype(int)

    tp = np.sum(
        poisoned_predictions == 1
    )

    fn = np.sum(
        poisoned_predictions == 0
    )

    fp = np.sum(
        clean_predictions == 1
    )

    tn = np.sum(
        clean_predictions == 0
    )

    total = (
        tp + tn + fp + fn
    )

    detection_accuracy = (
        (tp + tn) / total * 100
    )

    tpr = (
        tp / (tp + fn) * 100
        if (tp + fn) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn) * 100
        if (fp + tn) > 0
        else 0
    )

    return {
        "Detection Accuracy": detection_accuracy,
        "TPR": tpr,
        "FPR": fpr,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }