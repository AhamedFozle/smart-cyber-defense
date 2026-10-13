import numpy as np
import torch
from sklearn.ensemble import IsolationForest

def get_stratified_indices(
    dataset,
    num_clean=4500,
    num_poisoned=500,
    seed=42
):
    """
    Select an exact number of clean and poisoned samples.

    Returns:
        selected_indices: shuffled list of dataset indices
    """

    import random

    clean_indices = []
    poisoned_indices = []

    print("Finding clean and poisoned samples...")

    for index in range(len(dataset)):

        sample = dataset[index]

        # BadNetsDataset returns:
        # image, training_label, is_poisoned, original_label
        is_poisoned = sample[2]

        if is_poisoned:
            poisoned_indices.append(index)
        else:
            clean_indices.append(index)

    print(f"Available clean samples: {len(clean_indices)}")
    print(f"Available poisoned samples: {len(poisoned_indices)}")

    # Reproducible random selection
    random.seed(seed)

    selected_clean = random.sample(
        clean_indices,
        num_clean
    )

    selected_poisoned = random.sample(
        poisoned_indices,
        num_poisoned
    )

    selected_indices = (
        selected_clean +
        selected_poisoned
    )

    # Shuffle so clean and poisoned samples are mixed
    random.shuffle(selected_indices)

    print("\nStratified subset created")
    print(f"Clean selected: {len(selected_clean)}")
    print(f"Poisoned selected: {len(selected_poisoned)}")
    print(f"Total selected: {len(selected_indices)}")

    return selected_indices


def extract_cnn_features(
    model,
    dataset,
    device,
    indices=None
):
    """
    Extract CNN features.

    Args:
        model: Trained CNN model
        dataset: BadNets dataset
        device: CPU or CUDA
        indices: Specific dataset indices to process.
                 If None, all samples are used.

    Returns:
        features: CNN feature vectors
        true_labels: 1 = poisoned, 0 = clean
    """

    model.eval()

    features = []
    true_labels = []

    if indices is None:
        indices = list(range(len(dataset)))

    total_samples = len(indices)

    print(
        f"\nExtracting CNN features from "
        f"{total_samples} stratified samples..."
    )

    with torch.no_grad():

        for position, index in enumerate(indices):

            image, _, is_poisoned, _ = dataset[index]

            image = image.unsqueeze(0).to(device)

            feature = model.extract_features(image)

            feature = feature.cpu().numpy().flatten()

            features.append(feature)

            true_labels.append(
                1 if is_poisoned else 0
            )

            if (position + 1) % 1000 == 0:
                print(
                    f"Processed "
                    f"{position + 1}/{total_samples}"
                )

    features = np.array(
        features,
        dtype=np.float32
    )

    true_labels = np.array(
        true_labels
    )

    print("CNN feature extraction completed!")
    print(f"Feature shape: {features.shape}")

    return features, true_labels


def run_isolation_forest(
    features,
    contamination=0.10,
    random_state=42
):

    print("\nTraining Isolation Forest...")

    detector = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1
    )

    predictions = detector.fit_predict(features)

    # -1 = anomaly, 1 = normal
    predicted_labels = np.where(
        predictions == -1,
        1,
        0
    )

    print("Isolation Forest detection completed!")

    return predicted_labels


def calculate_detection_metrics(
    true_labels,
    predicted_labels
):

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    tp = np.sum(
        (true_labels == 1) &
        (predicted_labels == 1)
    )

    tn = np.sum(
        (true_labels == 0) &
        (predicted_labels == 0)
    )

    fp = np.sum(
        (true_labels == 0) &
        (predicted_labels == 1)
    )

    fn = np.sum(
        (true_labels == 1) &
        (predicted_labels == 0)
    )

    total = len(true_labels)

    detection_accuracy = (
        100.0 * (tp + tn) / total
    )

    tpr = (
        100.0 * tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    fpr = (
        100.0 * fp / (fp + tn)
        if (fp + tn) > 0 else 0.0
    )

    return {
         "accuracy": detection_accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }