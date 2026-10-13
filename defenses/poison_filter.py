from torch.utils.data import Subset


def create_filtered_dataset(
    dataset,
    selected_indices,
    predicted_labels
):
    """
    Remove samples predicted as suspicious.

    Args:
        dataset: Original BadNetsDataset
        selected_indices: Indices examined by the detector
        predicted_labels:
            1 = suspicious / remove
            0 = clean / keep

    Returns:
        filtered_dataset: Dataset containing retained samples
        removed_indices: Original dataset indices removed
    """

    if len(selected_indices) != len(predicted_labels):
        raise ValueError(
            "selected_indices and predicted_labels "
            "must have the same length."
        )

    suspicious_indices = set()

    for dataset_index, prediction in zip(
        selected_indices,
        predicted_labels
    ):
        if prediction == 1:
            suspicious_indices.add(dataset_index)

    all_indices = set(range(len(dataset)))

    kept_indices = sorted(
        all_indices - suspicious_indices
    )

    removed_indices = sorted(
        suspicious_indices
    )

    filtered_dataset = Subset(
        dataset,
        kept_indices
    )

    print("\n" + "=" * 55)
    print("POISON FILTERING SUMMARY")
    print("=" * 55)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Samples inspected: {len(selected_indices)}")
    print(f"Samples removed: {len(removed_indices)}")
    print(f"Samples retained: {len(filtered_dataset)}")
    print("=" * 55)

    return filtered_dataset, removed_indices