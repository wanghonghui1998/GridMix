import torch
import math

def gaussian_kernel(size: int, sigma: float):
    """
    Generates a 2D Gaussian kernel.

    Args:
        size (int): The size of the kernel. Should be an odd number.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel of shape (size, size).
    """
    # Ensure size is an odd number
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Create a coordinate grid of size x size
    ax = torch.arange(-size // 2 + 1, size // 2 + 1).float()
    xx, yy = torch.meshgrid(ax, ax)
    
    # Calculate the 2D Gaussian kernel
    kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    
    # Normalize the kernel to ensure the sum is 1
    kernel = kernel / torch.sum(kernel)
    
    return kernel

# Example usage
size = 3
sigma = 1.7

kernel = gaussian_kernel(size, sigma)
print("Gaussian kernel:")
print(kernel)
