import torch
import torch.nn.functional as F
import random
import pickle
from tqdm import tqdm

# Just some very quick example data. You want to replace this with your own version of course.


def generate_noise(shape, scale=1):
    h, w = shape
    noise = torch.zeros(h, w)

    # Mix multiple frequency bands
    for octave in range(4):
        freq = 2**octave
        grid_h, grid_w = h // (scale * freq), w // (scale * freq)

        # Generate random grid
        grid = torch.randn(grid_h + 1, grid_w + 1)

        # Upsample with bilinear interpolation
        upsampled = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        ).squeeze()

        noise += upsampled * (2**octave)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_displacement_field(image_size, flow_scale):
    """Generate smooth random displacement field."""
    h, w = image_size

    # Generate displacement on coarse grid
    coarse_size = (h // 32, w // 32)
    dx_coarse = torch.randn(1, 1, *coarse_size) * flow_scale
    dy_coarse = torch.randn(1, 1, *coarse_size) * flow_scale

    # Upsample to full resolution for smooth field
    dx = F.interpolate(dx_coarse, size=image_size, mode="bicubic", align_corners=True)
    dy = F.interpolate(dy_coarse, size=image_size, mode="bicubic", align_corners=True)

    # Shape: (1, 2, H, W)
    displacement = torch.cat([dx, dy], dim=1)
    return displacement.squeeze(0)  # (2, H, W)


def apply_displacement(image, displacement):
    """
    Apply displacement field to image using backwards warping.

    Args:
        image: (H, W) tensor
        displacement: (2, H, W) tensor with dx, dy

    Returns:
        warped_image: (H, W) tensor
    """
    h, w = image.shape

    assert image.shape == displacement.shape[1:]

    # Create base coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )

    # Apply displacement (backwards warping: sample from source)
    sample_x = grid_x - displacement[0]
    sample_y = grid_y - displacement[1]

    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (w - 1) - 1.0
    sample_y = 2.0 * sample_y / (h - 1) - 1.0

    # (1, H, W, 2)
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)

    # (1, 1, H, W)
    image_batch = image.unsqueeze(0).unsqueeze(0)

    warped = F.grid_sample(
        image_batch, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    # (H, W)
    return warped.squeeze()


def generate_data(image_size=(256, 256), flow_scale=2):
    img1 = generate_noise(image_size)

    # Generate displacement field
    displacement = generate_displacement_field(image_size, flow_scale)

    # Apply displacement to create second image
    img2 = apply_displacement(img1, displacement)

    # Flipping the two gives the same displacement but backwards
    if random.random() < 0.5:
        img1, img2 = img2, img1
        displacement = -displacement

    img = torch.stack([img1, img2], dim=0)

    return img, displacement


def generate_dataset(num_samples=500, out_file="dataset.pickle"):
    print(f"Generating {num_samples} samples...")

    dataset = []

    # Generate and save each sample
    for idx in tqdm(range(num_samples)):
        img, displacement = generate_data(image_size=image_size, flow_scale=flow_scale)

        sample = (img, displacement)
        dataset.append(sample)

    with open("dataset.pickle", "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved to {out_file}")


if __name__ == "__main__":
    image_size = (256, 256)
    num_samples = 500
    # Standard deviation of displacement field
    flow_scale = 2

    # Generate dataset
    generate_dataset(num_samples=num_samples)

    # Visualize example
    import matplotlib.pyplot as plt

    img, flow = generate_data(image_size=image_size, flow_scale=flow_scale)

    print()
    print(f"Flow range: [{flow.min():.2f}, {flow.max():.2f}]")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img[0].squeeze(), cmap="gray")
    axes[0, 0].set_title("Image 1")
    axes[0, 1].imshow(img[1].squeeze(), cmap="gray")
    axes[0, 1].set_title("Image 2 (displaced)")
    axes[1, 0].imshow(flow[0], cmap="RdBu")
    axes[1, 0].set_title("Displacement X")
    axes[1, 1].imshow(flow[1], cmap="RdBu")
    axes[1, 1].set_title("Displacement Y")
    plt.tight_layout()
    plt.show()
