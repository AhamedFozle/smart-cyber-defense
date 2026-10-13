import torch
import torch.nn.functional as F


def create_patch_mask(
    image_size=32,
    patch_size=3,
    device="cpu"
):
    """
    Create a fixed localized mask for a small trigger patch.

    The patch is placed in the bottom-right corner,
    matching the BadNets configuration used in this project.
    """

    mask = torch.zeros(
        1,
        1,
        image_size,
        image_size,
        device=device
    )

    mask[
        :,
        :,
        image_size - patch_size:image_size,
        image_size - patch_size:image_size
    ] = 1.0

    return mask


def apply_trigger(
    images,
    mask,
    pattern
):
    """
    Apply a localized trigger pattern.
    """

    return (
        images * (1.0 - mask)
        + pattern * mask
    )


def optimize_trigger(
    model,
    images,
    target_class,
    device,
    patch_size=3,
    iterations=100,
    learning_rate=0.05
):
    """
    Localized Neural Cleanse-style trigger optimization.

    The trigger is restricted to a small patch.
    """

    model.eval()

    image_size = images.shape[2]

    # Fixed localized trigger mask
    mask = create_patch_mask(
        image_size=image_size,
        patch_size=patch_size,
        device=device
    )

    # Learnable trigger pattern
    pattern = torch.zeros(
        1,
        3,
        image_size,
        image_size,
        device=device,
        requires_grad=True
    )

    optimizer = torch.optim.Adam(
        [pattern],
        lr=learning_rate
    )

    target = torch.full(
        (images.size(0),),
        target_class,
        dtype=torch.long,
        device=device
    )

    for iteration in range(iterations):

        optimizer.zero_grad()

        triggered_images = apply_trigger(
            images,
            mask,
            pattern
        )

        output = model(
            triggered_images
        )

        classification_loss = F.cross_entropy(
            output,
            target
        )

        classification_loss.backward()

        optimizer.step()

        # Keep pattern within valid normalized-image range
        with torch.no_grad():
            pattern.clamp_(-3.0, 3.0)

    # Final evaluation
    with torch.no_grad():

        triggered_images = apply_trigger(
            images,
            mask,
            pattern
        )

        output = model(
            triggered_images
        )

        predictions = torch.argmax(
            output,
            dim=1
        )

        attack_success_rate = (
            predictions == target
        ).float().mean().item() * 100

    # Trigger area
    mask_area = torch.mean(mask).item()

    return {
        "mask": mask.detach(),
        "pattern": pattern.detach(),
        "mask_norm": mask_area,
        "attack_success_rate": attack_success_rate
    }