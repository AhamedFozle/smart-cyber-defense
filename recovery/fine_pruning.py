import torch
import torch.nn as nn


class FinePruner:
    """
    CPU-friendly Fine-Pruning implementation.

    Steps:
    1. Measure average absolute activation of CNN channels.
    2. Select low-activation channels.
    3. Create binary pruning masks.
    4. Zero selected convolution filters.
    5. Reapply masks after optimizer updates so that
       pruned filters remain permanently disabled.
    """

    def __init__(self, model, device):

        self.model = model
        self.device = device

        self.conv_layers = []

        for name, module in model.named_modules():

            if isinstance(module, nn.Conv2d):

                self.conv_layers.append(
                    (name, module)
                )

    # ======================================================
    # COLLECT CHANNEL ACTIVATIONS
    # ======================================================

    def collect_channel_activations(
        self,
        data_loader
    ):
        """
        Collect average absolute activation
        for every convolutional channel.
        """

        activation_sums = {}
        activation_counts = {}

        hooks = []

        # --------------------------------------------------
        # Create hook
        # --------------------------------------------------

        def create_hook(name):

            def hook(
                module,
                inputs,
                output
            ):

                # Output shape:
                # [batch, channels, height, width]

                activation = output.detach()

                channel_mean = torch.mean(
                    torch.abs(activation),
                    dim=(0, 2, 3)
                )

                if name not in activation_sums:

                    activation_sums[name] = (
                        torch.zeros(
                            module.out_channels,
                            device=self.device
                        )
                    )

                    activation_counts[name] = 0

                activation_sums[name] += (
                    channel_mean
                )

                activation_counts[name] += 1

            return hook

        # --------------------------------------------------
        # Register hooks
        # --------------------------------------------------

        for name, layer in self.conv_layers:

            handle = layer.register_forward_hook(
                create_hook(name)
            )

            hooks.append(handle)

        # --------------------------------------------------
        # Evaluate model
        # --------------------------------------------------

        self.model.eval()

        print()
        print("Collecting channel activations...")

        with torch.no_grad():

            for batch_index, batch in enumerate(
                data_loader
            ):

                # BadNetsDataset returns:
                #
                # image
                # label
                # is_poisoned
                # original_label

                images = batch[0].to(
                    self.device
                )

                self.model(images)

                if (
                    (batch_index + 1) % 20
                    == 0
                ):

                    print(
                        f"Processed batches: "
                        f"{batch_index + 1}"
                    )

        # --------------------------------------------------
        # Remove hooks
        # --------------------------------------------------

        for handle in hooks:

            handle.remove()

        # --------------------------------------------------
        # Calculate average activations
        # --------------------------------------------------

        average_activations = {}

        for name in activation_sums:

            average_activations[name] = (
                activation_sums[name]
                / activation_counts[name]
            )

        print(
            "Activation collection completed."
        )

        return average_activations

    # ======================================================
    # CREATE PRUNING MASKS
    # ======================================================

    def create_pruning_masks(
        self,
        average_activations,
        pruning_rate=0.10
    ):
        """
        Select the lowest-activation channels.

        Mask:
            1 = keep channel
            0 = prune channel
        """

        masks = {}

        print()
        print("Creating pruning masks...")

        for name, layer in self.conv_layers:

            activations = (
                average_activations[name]
            )

            num_channels = (
                layer.out_channels
            )

            # Number of channels to prune
            num_prune = max(
                1,
                int(
                    num_channels
                    * pruning_rate
                )
            )

            # Find lowest activation channels
            _, indices = torch.topk(
                activations,
                k=num_prune,
                largest=False
            )

            # Start with all channels active
            mask = torch.ones(
                num_channels,
                device=self.device
            )

            # Disable selected channels
            mask[indices] = 0.0

            masks[name] = mask

            print(
                f"{name}: "
                f"{num_prune}/{num_channels} "
                f"channels selected for pruning"
            )

        return masks

    # ======================================================
    # APPLY MASKS
    # ======================================================

    def apply_masks(
        self,
        masks
    ):
        """
        Apply pruning masks to convolution filters.

        Selected filters and biases are set to zero.
        """

        print()
        print("Applying pruning masks...")

        with torch.no_grad():

            for name, layer in self.conv_layers:

                mask = masks[name]

                # Convert:
                #
                # [channels]
                #
                # to:
                #
                # [channels, 1, 1, 1]

                weight_mask = mask.view(
                    -1,
                    1,
                    1,
                    1
                )

                # Zero selected filters
                layer.weight.mul_(
                    weight_mask
                )

                # Zero corresponding biases
                if layer.bias is not None:

                    layer.bias.mul_(
                        mask
                    )

        print(
            "Pruning masks applied successfully."
        )

    # ======================================================
    # REAPPLY MASKS
    # ======================================================

    def reapply_masks(
        self,
        masks
    ):
        """
        Reapply pruning masks after every optimizer step.

        This prevents the optimizer from restoring
        previously pruned filters.
        """

        with torch.no_grad():

            for name, layer in self.conv_layers:

                mask = masks[name]

                weight_mask = mask.view(
                    -1,
                    1,
                    1,
                    1
                )

                # Keep pruned filters at zero
                layer.weight.mul_(
                    weight_mask
                )

                # Keep pruned biases at zero
                if layer.bias is not None:

                    layer.bias.mul_(
                        mask
                    )

    # ======================================================
    # COUNT PRUNED CHANNELS
    # ======================================================

    def count_pruned_channels(
        self,
        masks
    ):

        total_channels = 0
        pruned_channels = 0

        for name, layer in self.conv_layers:

            total_channels += (
                layer.out_channels
            )

            pruned_channels += int(
                torch.sum(
                    masks[name] == 0
                ).item()
            )

        return (
            pruned_channels,
            total_channels
        )

    # ======================================================
    # VERIFY PRUNED FILTERS
    # ======================================================

    def verify_masks(
        self,
        masks
    ):
        """
        Check that all pruned filters are zero.
        """

        maximum_pruned_weight = 0.0

        with torch.no_grad():

            for name, layer in self.conv_layers:

                mask = masks[name]

                pruned_indices = (
                    mask == 0
                )

                if (
                    pruned_indices.sum().item()
                    == 0
                ):

                    continue

                pruned_weights = (
                    layer.weight[
                        pruned_indices
                    ]
                )

                if pruned_weights.numel() > 0:

                    current_max = torch.max(
                        torch.abs(
                            pruned_weights
                        )
                    ).item()

                    maximum_pruned_weight = max(
                        maximum_pruned_weight,
                        current_max
                    )

        return maximum_pruned_weight

    # ======================================================
    # COMPLETE PRUNING PIPELINE
    # ======================================================

    def prune(
        self,
        data_loader,
        pruning_rate=0.10
    ):
        """
        Complete fine-pruning pipeline.
        """

        # --------------------------------------------------
        # Step 1: Collect activations
        # --------------------------------------------------

        average_activations = (
            self.collect_channel_activations(
                data_loader
            )
        )

        # --------------------------------------------------
        # Step 2: Create masks
        # --------------------------------------------------

        masks = (
            self.create_pruning_masks(
                average_activations,
                pruning_rate
            )
        )

        # --------------------------------------------------
        # Step 3: Apply masks
        # --------------------------------------------------

        self.apply_masks(
            masks
        )

        # --------------------------------------------------
        # Step 4: Count channels
        # --------------------------------------------------

        pruned, total = (
            self.count_pruned_channels(
                masks
            )
        )

        # --------------------------------------------------
        # Print summary
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("FINE-PRUNING SUMMARY")
        print("=" * 60)

        print(
            f"Pruned channels: "
            f"{pruned}/{total}"
        )

        print(
            f"Pruning percentage: "
            f"{100.0 * pruned / total:.2f}%"
        )

        # --------------------------------------------------
        # Verify masks
        # --------------------------------------------------

        maximum_weight = (
            self.verify_masks(
                masks
            )
        )

        print(
            f"Maximum pruned-filter weight: "
            f"{maximum_weight:.10f}"
        )

        return masks