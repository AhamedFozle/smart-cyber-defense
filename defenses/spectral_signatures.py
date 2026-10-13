import numpy as np
import torch


def extract_features(
    model,
    dataset,
    device,
    indices=None
):
    """
    Extract learned CNN features for selected samples.

    Returns:
        features: CNN feature vectors
        true_labels: 1 = poisoned, 0 = clean

    Note:
        true_labels are used only for evaluation.
    """

    model.eval()

    features = []
    true_labels = []

    if indices is None:
        indices = list(range(len(dataset)))

    total_samples = len(indices)

    print(
        f"\nExtracting features from "
        f"{total_samples} samples..."
    )

    with torch.no_grad():

        for position, index in enumerate(indices):

            image, _, is_poisoned, _ = dataset[index]

            image = image.unsqueeze(0).to(device)

            feature = model.extract_features(image)

            feature = (
                feature
                .cpu()
                .numpy()
                .flatten()
            )

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
        true_labels,
        dtype=np.int64
    )

    print(
        "Feature extraction completed!"
    )

    print(
        f"Feature shape: "
        f"{features.shape}"
    )

    return features, true_labels


def fit_spectral_direction(
    reference_features
):
    """
    Learn the dominant spectral direction from
    a reference dataset.

    The reference dataset should contain clean
    samples.

    Returns:
        reference_mean
        dominant_direction
    """

    reference_features = np.asarray(
        reference_features,
        dtype=np.float64
    )

    # Center reference features
    reference_mean = np.mean(
        reference_features,
        axis=0,
        keepdims=True
    )

    centered_reference = (
        reference_features
        - reference_mean
    )

    print(
        "Computing SVD on clean reference data..."
    )

    _, _, vh = np.linalg.svd(
        centered_reference,
        full_matrices=False
    )

    dominant_direction = vh[0]

    return (
        reference_mean,
        dominant_direction
    )


def calculate_spectral_scores(
    features,
    reference_mean,
    dominant_direction
):
    """
    Calculate spectral outlier scores using
    a previously learned reference direction.

    IMPORTANT:
    The same reference mean and dominant direction
    are used for all evaluation samples.
    """

    features = np.asarray(
        features,
        dtype=np.float64
    )

    # Center using the CLEAN reference mean
    centered_features = (
        features
        - reference_mean
    )

    # Project onto reference spectral direction
    projections = np.dot(
        centered_features,
        dominant_direction
    )

    # Squared projection = spectral score
    scores = projections ** 2

    return scores


def run_spectral_signatures(
    features,
    num_poisoned=500
):
    """
    Original controlled Spectral Signatures experiment.

    This version assumes the number of suspicious
    samples is known.

    It is retained as the controlled baseline.
    """

    print(
        "\nRunning Spectral Signatures "
        "(known suspicious count)..."
    )

    features = np.asarray(
        features,
        dtype=np.float64
    )

    centered_features = (
        features
        - np.mean(
            features,
            axis=0,
            keepdims=True
        )
    )

    print("Computing SVD...")

    _, _, vh = np.linalg.svd(
        centered_features,
        full_matrices=False
    )

    dominant_direction = vh[0]

    projections = np.dot(
        centered_features,
        dominant_direction
    )

    scores = projections ** 2

    n_samples = len(features)

    num_poisoned = min(
        num_poisoned,
        n_samples
    )

    suspicious_indices = np.argsort(
        scores
    )[-num_poisoned:]

    predicted_labels = np.zeros(
        n_samples,
        dtype=np.int64
    )

    predicted_labels[
        suspicious_indices
    ] = 1

    print(
        "Spectral Signatures completed!"
    )

    print(
        f"Samples flagged as suspicious: "
        f"{len(suspicious_indices)}"
    )

    print(
        f"Highest score: "
        f"{scores.max():.6f}"
    )

    return predicted_labels, scores


def run_spectral_threshold_detection(
    reference_features,
    evaluation_features,
    percentile=99.0
):
    """
    Label-independent threshold-based
    Spectral Signatures experiment.

    The spectral direction is learned ONLY from
    clean reference features.

    The detection threshold is also calculated
    ONLY from clean reference scores.

    The evaluation dataset is then classified
    without using poison labels.

    Returns:
        predicted_labels
        evaluation_scores
        threshold
    """

    print(
        "\nRunning Threshold-Based "
        "Spectral Signatures..."
    )

    # ==================================================
    # 1. Learn spectral direction from clean reference
    # ==================================================

    reference_mean, dominant_direction = (
        fit_spectral_direction(
            reference_features
        )
    )

    # ==================================================
    # 2. Calculate scores for clean reference data
    # ==================================================

    reference_scores = (
        calculate_spectral_scores(
            reference_features,
            reference_mean,
            dominant_direction
        )
    )

    # ==================================================
    # 3. Calculate threshold
    #
    # No poison labels are used.
    # ==================================================

    threshold = np.percentile(
        reference_scores,
        percentile
    )

    # ==================================================
    # 4. Score evaluation samples using the SAME
    #    reference direction
    # ==================================================

    evaluation_scores = (
        calculate_spectral_scores(
            evaluation_features,
            reference_mean,
            dominant_direction
        )
    )

    # ==================================================
    # 5. Predict suspicious samples
    # ==================================================

    predicted_labels = (
        evaluation_scores >= threshold
    ).astype(np.int64)

    print(
        "Threshold-based detection completed!"
    )

    print(
        f"Reference percentile: "
        f"{percentile:.2f}%"
    )

    print(
        f"Detection threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Samples flagged as suspicious: "
        f"{np.sum(predicted_labels)}"
    )

    print(
        f"Highest evaluation score: "
        f"{evaluation_scores.max():.6f}"
    )

    return (
        predicted_labels,
        evaluation_scores,
        threshold
    )