import torch
from models.cnn import SimpleCNN


# Create the model
model = SimpleCNN()

# Create a fake batch of CIFAR-10 images
sample_images = torch.randn(64, 3, 32, 32)

# Pass images through the model
output = model(sample_images)

print("CNN model created successfully!")
print(f"Input shape: {sample_images.shape}")
print(f"Output shape: {output.shape}")

# Number of parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Total trainable parameters: {total_parameters:,}")