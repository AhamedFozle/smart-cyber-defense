from torchvision import datasets, transforms

from attacks.badnets import BadNetsDataset


# Transform CIFAR-10 images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])


# Load the original CIFAR-10 training dataset
clean_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Create the poisoned dataset
poisoned_dataset = BadNetsDataset(
    dataset=clean_dataset,
    poison_rate=0.1,
    target_class=0,
    trigger_size=3,
    trigger_value=1.0,
    seed=42
)


print("\nChecking dataset samples...")
print("-" * 40)

poisoned_count = 0
clean_count = 0


# Check all samples
for index in range(len(poisoned_dataset)):

    image, label, is_poisoned, original_label = poisoned_dataset[index]

    if is_poisoned:
        poisoned_count += 1
    else:
        clean_count += 1


print(f"Total samples checked: {len(poisoned_dataset)}")
print(f"Poisoned samples found: {poisoned_count}")
print(f"Clean samples found: {clean_count}")

print("\nChecking example samples...")
print("-" * 40)


# Find one poisoned sample
for index in range(len(poisoned_dataset)):

    image, label, is_poisoned, original_label = poisoned_dataset[index]

    if is_poisoned:
        print(f"Dataset index: {index}")
        print(f"Image shape: {image.shape}")
        print(f"Is poisoned: {is_poisoned}")
        print(f"Original label: {original_label}")
        print(f"Training label: {label}")
        break


print("\nPoisoned dataset test completed successfully!")