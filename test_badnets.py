import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from attacks.badnets import add_trigger


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Get one clean image
image, label = dataset[0]

# Add trigger
triggered_image = add_trigger(
    image,
    trigger_size=3,
    trigger_value=1.0
)

print("BadNets trigger test successful!")
print(f"Original image shape: {image.shape}")
print(f"Triggered image shape: {triggered_image.shape}")
print(f"Original label: {label}")

# Convert PyTorch tensors from [C, H, W] to [H, W, C]
clean_image = image.permute(1, 2, 0)
poisoned_image = triggered_image.permute(1, 2, 0)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(clean_image)
axes[0].set_title("Clean Image")
axes[0].axis("off")

axes[1].imshow(poisoned_image)
axes[1].set_title("Triggered Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("results/badnets_trigger_example.png", dpi=150)

print("Trigger image saved to:")
print("results/badnets_trigger_example.png")

plt.show()