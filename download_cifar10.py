from torchvision import datasets, transforms

# Convert images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

print("Downloading CIFAR-10 training dataset...")

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("Downloading CIFAR-10 test dataset...")

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("\nCIFAR-10 downloaded successfully!")
print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"Classes: {classes}")

# Check one sample
image, label = train_dataset[0]

print("\nSample image information:")
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print(f"Class: {classes[label]}")