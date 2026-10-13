from utils.data_loader import get_data_loaders, get_classes


train_loader, test_loader = get_data_loaders()

images, labels = next(iter(train_loader))

print("Data loader working successfully!")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")
print(f"Classes: {get_classes()}")