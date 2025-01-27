import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  
#     transforms.ToTensor(),         
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
# ])


transform = transforms.Compose([transforms.ToTensor()])

data_dir = "./data"
cifar10_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)

# Number of rows and columns
rows, cols = 8, 5

# Create a figure with a specific size
plt.figure(figsize=(10, 10))  # Adjust the figure size for better visibility

# Plot 25 images (5x5 grid)
for i in range(rows * cols):
    plt.subplot(rows, cols, i + 1)  # Specify the grid (rows, cols, index)
    plt.imshow(cifar10_dataset.data[i])  # Plot the image
    plt.axis('off')  # Turn off axes
    plt.title(f"Label: {cifar10_dataset.targets[i]}", fontsize=8)  # Add label as title

# Adjust layout
plt.tight_layout()
plt.show()
