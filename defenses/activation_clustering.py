import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def extract_activations(
    model,
    dataset,
    device,
    indices=None
):
    """
    Extract CNN activations for Activation Clustering.

    Returns:
        activations: CNN feature vectors
        true_labels: 1 = poisoned, 0 = clean

    Note:
        true_labels are returned ONLY for evaluation.
        They are NOT used to determine the suspicious cluster.
    """

    model.eval()

    activations = []
    true_labels = []

    if indices is None:
        indices = list(range(len(dataset)))

    total_samples = len(indices)

    print(
        f"\nExtracting activations from "
        f"{total_samples} samples..."
    )

    with torch.no_grad():

        for position, index in enumerate(indices):

            image, _, is_poisoned, _ = dataset[index]

            image = image.unsqueeze(0).to(device)

            activation = model.extract_features(image)

            activation = (
                activation
                .cpu()
                .numpy()
                .flatten()
            )

            activations.append(activation)

            # Used only after clustering for evaluation
            true_labels.append(
                1 if is_poisoned else 0
            )

            if (position + 1) % 1000 == 0:

                print(
                    f"Processed "
                    f"{position + 1}/{total_samples}"
                )

    activations = np.array(
        activations,
        dtype=np.float32
    )

    true_labels = np.array(
        true_labels,
        dtype=np.int64
    )

    print(
        "Activation extraction completed!"
    )

    print(
        f"Activation shape: "
        f"{activations.shape}"
    )

    return activations, true_labels


def run_activation_clustering(
    activations,
    true_labels=None,
    random_state=42
):
    """
    Perform label-independent Activation Clustering.

    K-Means creates two clusters.

    IMPORTANT:
    True poison/clean labels are NOT used to select
    the suspicious cluster.

    The suspicious cluster is selected using an
    unsupervised size-based criterion: the smaller
    cluster is treated as suspicious.

    This avoids label leakage.

    true_labels is accepted only so that the function
    can calculate evaluation metrics after clustering.

    Returns:
        predicted_labels
        cluster_info
    """

    print(
        "\nRunning Label-Independent "
        "Activation Clustering..."
    )

    # --------------------------------------------------
    # K-Means clustering
    # --------------------------------------------------

    kmeans = KMeans(
        n_clusters=2,
        random_state=random_state,
        n_init=10
    )

    cluster_assignments = kmeans.fit_predict(
        activations
    )

    # --------------------------------------------------
    # Determine cluster sizes
    # --------------------------------------------------

    cluster_0_size = np.sum(
        cluster_assignments == 0
    )

    cluster_1_size = np.sum(
        cluster_assignments == 1
    )

    print(
        f"Cluster 0 size: "
        f"{cluster_0_size}"
    )

    print(
        f"Cluster 1 size: "
        f"{cluster_1_size}"
    )

    # --------------------------------------------------
    # Label-independent suspicious cluster selection
    #
    # We assume the poisoned samples form the smaller
    # and more compact cluster.
    #
    # NO true labels are used here.
    # --------------------------------------------------

    if cluster_0_size < cluster_1_size:

        suspicious_cluster = 0

    else:

        suspicious_cluster = 1

    print(
        f"Suspicious cluster selected: "
        f"{suspicious_cluster}"
    )

    # --------------------------------------------------
    # Convert cluster assignment into detection labels
    #
    # 1 = predicted poisoned
    # 0 = predicted clean
    # --------------------------------------------------

    predicted_labels = np.where(
        cluster_assignments == suspicious_cluster,
        1,
        0
    )

    # --------------------------------------------------
    # Evaluation
    #
    # True labels are used ONLY here.
    # --------------------------------------------------

    cluster_info = {
        "cluster_assignments": cluster_assignments,
        "suspicious_cluster": int(
            suspicious_cluster
        ),
        "cluster_0_size": int(
            cluster_0_size
        ),
        "cluster_1_size": int(
            cluster_1_size
        )
    }

    if true_labels is not None:

        accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        cluster_info["accuracy"] = (
            accuracy * 100.0
        )

        print(
            f"Detection accuracy: "
            f"{accuracy * 100:.2f}%"
        )

    return predicted_labels, cluster_info