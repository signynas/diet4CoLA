import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import argparse

# --- Configuration ---
WIDTH = 512
HEIGHT = 512
FPS = 10  # Frames per second for the output video

def visualize_cortex(file_path: str, output_dir: str):
    """Loads a 3D NPY file and saves it as an MP4 video."""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load the Data
    try:
        # The data is expected to be (T, H, W)
        advected_cortex = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    T, H, W = advected_cortex.shape
    print(f"Loaded simulation with {T} timepoints, size {H}x{W}.")

    # 2. Setup Plotting Environment
    # We close the figure at the end so it doesn't pop up if running non-interactively
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100) # Use DPI to control size
    
    # Use vmin/vmax based on the normalized cortex data (0 to 1)
    im = ax.imshow(advected_cortex[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{file_path.stem} | Time: 0 / {T-1}")
    ax.axis('off')

    # 3. Define the Animation Update Function
    def update_frame(t):
        """Update the image data for each frame."""
        im.set_data(advected_cortex[t])
        ax.set_title(f"{file_path.stem} | Time: {t} / {T-1}")
        return [im]

    # 4. Create the Animation object
    animation = FuncAnimation(
        fig,
        update_frame,
        frames=T,
        interval=1000/FPS, # Interval in milliseconds
        blit=True
    )
    
    # 5. Save as Video (Requires FFMpeg installed on your system)
    output_video_path = output_dir / f'{file_path.stem}.mp4'
    
    # Configure writer
    writer = FFMpegWriter(fps=FPS, bitrate=1800)
    
    print(f"Saving video to: {output_video_path.resolve()}...")
    animation.save(output_video_path, writer=writer)
    print("Video saved successfully.")
    
    plt.close(fig) # Clean up the figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Advected Cortex NPY File")
    parser.add_argument("--file", type=str, required=True, help="Path to the cortex_ID_advected.npy file")
    parser.add_argument("--out", type=str, default="./data", help="Output directory for the MP4 video")
    args = parser.parse_args()
    
    visualize_cortex(args.file, args.out)