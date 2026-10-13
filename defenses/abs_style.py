import torch
import torch.nn as nn


def get_conv_layers(model):
    """
    Return all convolutional layers in the model.
    """
    conv_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    return conv_layers


def get_baseline_logits(model, images, device):
    """
    Get model logits for clean images.
    """
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        logits = model(images)

    return logits


def stimulate_channel(
    model,
    layer,
    channel_index,
    images,
    device,
    stimulation_strength
):
    """
    Stimulate one convolutional channel and measure
    the resulting model output.

    This is an ABS-style prototype, not a full reproduction
    of the original ABS algorithm.
    """

    stimulated_output = None

    def hook(module, input, output):
        nonlocal stimulated_output

        modified = output.clone()

        modified[:, channel_index, :, :] += stimulation_strength

        stimulated_output = modified

        return modified

    handle = layer.register_forward_hook(hook)

    model.eval()

    with torch.no_grad():
        images = images.to(device)
        logits = model(images)

    handle.remove()

    return logits


def analyze_layer(
    model,
    layer_name,
    layer,
    images,
    device,
    stimulation_factor=2.0
):
    """
    Analyze every channel in one convolutional layer.

    For every channel:
    1. Measure its average activation.
    2. Stimulate the channel.
    3. Measure the change in class logits.
    4. Find which class is affected most.
    """

    print(f"\nAnalyzing layer: {layer_name}")
    print(f"Number of channels: {layer.out_channels}")

    model.eval()

    # --------------------------------------------------
    # Step 1: collect baseline activation statistics
    # --------------------------------------------------

    activation_storage = {}

    def activation_hook(module, input, output):
        activation_storage["activation"] = output.detach()

    handle = layer.register_forward_hook(activation_hook)

    with torch.no_grad():
        model(images.to(device))

    handle.remove()

    activations = activation_storage["activation"]

    # Average absolute activation for every channel
    channel_means = torch.mean(
        torch.abs(activations),
        dim=(0, 2, 3)
    )

    # --------------------------------------------------
    # Step 2: baseline logits
    # --------------------------------------------------

    baseline_logits = get_baseline_logits(
        model,
        images,
        device
    )

    baseline_mean = torch.mean(
        baseline_logits,
        dim=0
    )

    results = []

    # --------------------------------------------------
    # Step 3: stimulate every channel
    # --------------------------------------------------

    for channel_index in range(layer.out_channels):

        channel_activation = channel_means[channel_index].item()

        # Avoid zero stimulation
        stimulation_strength = max(
            channel_activation * stimulation_factor,
            0.1
        )

        stimulated_logits = stimulate_channel(
            model=model,
            layer=layer,
            channel_index=channel_index,
            images=images,
            device=device,
            stimulation_strength=stimulation_strength
        )

        stimulated_mean = torch.mean(
            stimulated_logits,
            dim=0
        )

        # Change in logits
        logit_change = stimulated_mean - baseline_mean

        # Find class most strongly affected
        strongest_class = torch.argmax(
            logit_change
        ).item()

        strongest_change = torch.max(
            logit_change
        ).item()

        results.append({
            "layer": layer_name,
            "channel": channel_index,
            "activation_mean": channel_activation,
            "stimulation": stimulation_strength,
            "associated_class": strongest_class,
            "logit_change": strongest_change
        })

        print(
            f"Channel {channel_index:3d} | "
            f"Class {strongest_class} | "
            f"Score {strongest_change:.4f}"
        )

    return results


def run_abs_style(
    model,
    images,
    device,
    stimulation_factor=2.0,
    top_k=10
):
    """
    Run ABS-style analysis across all convolutional layers.

    Returns the highest-ranked suspicious channels.
    """

    conv_layers = get_conv_layers(model)

    print("\n" + "=" * 60)
    print("ABS-STYLE BACKDOOR ANALYSIS")
    print("=" * 60)

    print(f"Convolutional layers found: {len(conv_layers)}")

    all_results = []

    for layer_name, layer in conv_layers:

        layer_results = analyze_layer(
            model=model,
            layer_name=layer_name,
            layer=layer,
            images=images,
            device=device,
            stimulation_factor=stimulation_factor
        )

        all_results.extend(layer_results)

    # --------------------------------------------------
    # Rank suspicious neurons/channels
    # --------------------------------------------------

    all_results.sort(
        key=lambda x: x["logit_change"],
        reverse=True
    )

    top_results = all_results[:top_k]

    print("\n" + "=" * 60)
    print(f"TOP {top_k} SUSPICIOUS CHANNELS")
    print("=" * 60)

    for rank, result in enumerate(top_results, start=1):

        print(
            f"{rank:2d}. "
            f"Layer={result['layer']} | "
            f"Channel={result['channel']} | "
            f"Class={result['associated_class']} | "
            f"Score={result['logit_change']:.4f}"
        )

    return top_results, all_results