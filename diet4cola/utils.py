import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_2d_array(data: np.ndarray, 
                  title: str = 'Noise',
                  cmap: str = 'gray') -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    img = data
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'{title}')

    plt.show()

def plot_2d_array_comparison(left: np.ndarray,
                             right: np.ndarray, 
                             left_title: str = 'Left', 
                             right_title: str = 'Right',
                             min: float = 0,
                             max: float = 1) -> None:
    cols = 2

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes = np.array(axes).reshape(1, cols)

    img_a = left
    axes[0, 0].imshow(img_a, cmap='gray', vmin=min, vmax=max)
    axes[0, 0].set_title(f'{left_title}')

    img_b = right
    axes[0, 1].imshow(img_b, cmap='gray', vmin=min, vmax=max)
    axes[0, 1].set_title(f'{right_title}')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.show()

def plot_synthetic_cut(data: np.ndarray,
                       cut_center: tuple[int, int],
                       cut_origin: tuple[int, int],
                       cut_destination: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    img = data
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Synthetic Cut')
    
    # Plot the cut line in green
    ax.plot(
        [cut_origin[0], cut_destination[0]], 
        [cut_origin[1], cut_destination[1]], 
        color='green', linewidth=2, label='Cut Line', zorder=1
    )

    # Plot the cut center in red
    ax.scatter(cut_center[0], cut_center[1], color='red', s=20, label='Cut Center', zorder=2)
    
    # Optional: add legend
    ax.legend(loc='upper right', fontsize=8)
    
    plt.show()

def plot_sdf(data: np.ndarray,
             origin: tuple[int, int],
             destination: tuple[int, int],
             title: str = 'SDF',
             cmap: str = 'RdBu_r') -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(data, cmap=cmap, origin='upper')
    ax.set_title(title)
    
    # Plot the origin–destination line in green
    ax.plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )

    # Plot origin and destination points in red
    ax.scatter(*origin, color='red', s=20, zorder=2)
    ax.scatter(*destination, color='red', s=20, zorder=2)
    
    # Overlay zero-contour
    ax.contour(data, levels=[0], colors='yellow', linewidths=1.5)

    # --- Add a colorbar that stays aligned with image height ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)
    
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_velocity_field(velocity: np.ndarray,
                        sdf_base: np.ndarray,
                        origin: tuple[int, int],
                        destination: tuple[int, int],
                        title: str = 'Velocity Field',
                        cmap: str = 'viridis'):
    cols = 2

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes = np.array(axes).reshape(1, cols)

    img_a = velocity
    img_a_mappable = axes[0, 0].imshow(img_a, cmap=cmap)
    axes[0, 0].set_title(f'{title}')

    # Plot the origin–destination line in green
    axes[0, 0].plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )

    # Plot origin and destination points in red
    axes[0, 0].scatter(*origin, color='red', s=20, zorder=2)
    axes[0, 0].scatter(*destination, color='red', s=20, zorder=2)

    # --- Add a colorbar that stays aligned with image height ---
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img_a_mappable, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)
    
    axes[0, 0].legend(loc='upper right', fontsize=8)

    img_b = sdf_base
    img_b_mappable = axes[0, 1].imshow(img_b, cmap='RdBu_r')
    axes[0, 1].set_title(f'SDF Base')

    # Plot the origin–destination line in green
    axes[0, 1].plot(
        [origin[0], destination[0]], 
        [origin[1], destination[1]], 
        color='green', linewidth=2, label='CoLA Cut', zorder=1
    )

    # Plot origin and destination points in red
    axes[0, 1].scatter(*origin, color='red', s=20, zorder=2)
    axes[0, 1].scatter(*destination, color='red', s=20, zorder=2)

    # Overlay zero-contour
    axes[0, 1].contour(sdf_base, levels=[0], colors='yellow', linewidths=1.5)

    # --- Add a colorbar that stays aligned with image height ---
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img_b_mappable, cax=cax)
    cbar.set_label('Distance (Euclidean)', rotation=270, labelpad=12)

    axes[0, 1].legend(loc='upper right', fontsize=8)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.show()

def save_cortex(data: np.ndarray, filename: str) -> None:
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if data.ndim != 2:
        raise ValueError("Array must be 2D.")
    np.save(filename, data)

def load_cortex(filename: str) -> np.ndarray:
    data = np.load(filename, allow_pickle=False)
    
    if data.ndim != 2:
        raise ValueError("Loaded data is not a 2D array.")
    return data