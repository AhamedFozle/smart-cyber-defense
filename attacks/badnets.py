
import torch
from torch.utils.data import Dataset


def add_trigger(image, trigger_size=4, trigger_value=1.0):
    """
    Add a square BadNets trigger to the bottom-right corner.

    Supports:
        Single image: [C, H, W]
        Batch of images: [B, C, H, W]

    Parameters:
        image: PyTorch tensor
        trigger_size: Size of the square trigger
        trigger_value: Pixel value used for the trigger

    Returns:
        Image tensor containing the trigger.
    """

    poisoned_image = image.clone()

    # --------------------------------------------------------
    # Single image: [C, H, W]
    # --------------------------------------------------------
    if poisoned_image.dim() == 3:

        _, height, width = poisoned_image.shape

        poisoned_image[
            :,
            height - trigger_size:height,
            width - trigger_size:width
        ] = trigger_value

    # --------------------------------------------------------
    # Batch of images: [B, C, H, W]
    # --------------------------------------------------------
    elif poisoned_image.dim() == 4:

        _, _, height, width = poisoned_image.shape

        poisoned_image[
            :,
            :,
            height - trigger_size:height,
            width - trigger_size:width
        ] = trigger_value

    # --------------------------------------------------------
    # Invalid shape
    # --------------------------------------------------------
    else:

        raise ValueError(
            f"Expected a 3D or 4D tensor, "
            f"but got shape {poisoned_image.shape}"
        )

    return poisoned_image


class BadNetsDataset(Dataset):
    """
    CIFAR-10 dataset wrapper for a controlled BadNets experiment.

    A selected percentage of training samples receive:
        1. A visible square trigger.
        2. The target-class label.

    Clean samples retain their original labels.
    """

    def __init__(
        self,
        dataset,
        poison_rate=0.1,
        target_class=0,
        trigger_size=3,
        trigger_value=1.0,
        seed=42
    ):

        self.dataset = dataset
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        # ----------------------------------------------------
        # Calculate number of poisoned samples
        # ----------------------------------------------------

        total_samples = len(dataset)

        num_poisoned = int(
            total_samples * poison_rate
        )

        # ----------------------------------------------------
        # Reproducible random selection
        # ----------------------------------------------------

        generator = torch.Generator()

        generator.manual_seed(seed)

        poisoned_indices = torch.randperm(
            total_samples,
            generator=generator
        )[:num_poisoned].tolist()

        self.poisoned_indices = set(
            poisoned_indices
        )

        # ----------------------------------------------------
        # Information about the experiment
        # ----------------------------------------------------

        print()
        print("BadNets Dataset Created")
        print("-" * 40)

        print(
            f"Total samples:    {total_samples}"
        )

        print(
            f"Poisoned samples: {num_poisoned}"
        )

        print(
            f"Poison rate:      {poison_rate * 100:.1f}%"
        )

        print(
            f"Target class:     {target_class}"
        )

        print("-" * 40)

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        image, original_label = (
            self.dataset[index]
        )

        # ----------------------------------------------------
        # Determine whether this sample is poisoned
        # ----------------------------------------------------

        is_poisoned = (
            index in self.poisoned_indices
        )

        # ----------------------------------------------------
        # Poisoned sample
        # ----------------------------------------------------

        if is_poisoned:

            image = add_trigger(
                image,
                trigger_size=self.trigger_size,
                trigger_value=self.trigger_value
            )

            label = self.target_class

        # ----------------------------------------------------
        # Clean sample
        # ----------------------------------------------------

        else:

            label = original_label

        # ----------------------------------------------------
        # Return metadata
        #
        # image
        # training label
        # whether poisoned
        # original label
        # ----------------------------------------------------

        return (
            image,
            label,
            is_poisoned,
            original_label
        )

