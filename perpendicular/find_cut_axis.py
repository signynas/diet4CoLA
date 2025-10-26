# Load modules
import tqdm
import czifile
import tifffile
import sys
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
from IPython.display import Image as IPImage, display
from matplotlib.animation import FuncAnimation
from skimage.io import imread
from itertools import combinations
import cv2

class FindCut:
    """
    A class for finding the cut caused by Cortical Laser Ablation in a microscopy video.
    """

    def __init__(self, experiment: str, point_file: str, data_dir: str = '../../data', output_path: str = 'out/', frame_idx: int = 4, color_channel: int = 0, cmap: str = 'gray', interpolation_method: str = cv2.INTER_CUBIC):
        self.experiment = experiment
        self.point_file = point_file
        self.data_dir = data_dir
        self.output_path = output_path
        self.frame_idx = frame_idx
        self.color_channel = color_channel
        self.cmap = cmap
        self.interpolation_method = interpolation_method

        if not len(self.experiment) > 0 or not isinstance(self.experiment, str):
            raise ValueError("Experiment name is invalid.")
        
        if not isinstance(self.frame_idx, int):
            raise ValueError('Frame number has to be an integer.')

        if not self.cmap in set('gray', 'viridis', 'magma'):
            raise ValueError('Cmap has to be one of the following: gray, viridis, magma.')
        
        if not self.color_channel in set(0, 1):
            raise ValueError('Color channel has to be either 0 (red), or 1 (green).')
        
        # Convert images
        self._convert_images_to_tiff()
        self.image_dir = os.path.join(os.path.join(self.data_dir, 'tiff'))

        # Read in image
        self.image, self.image_data, self.image_shape = self._read_image()

        # Prepare for plotting
        self.rgb_stack = self._prepare_plotting()

    def __repr__(self) -> str:
        # UPDATE!
        pass
    
    def _convert_images_to_tiff(self):
        """Convert czi images to tiff."""
        # Declare folders
        src_folder = os.path.join(self.data_dir, 'ablation-czi') 
        new_folder = os.path.join(self.data_dir, 'tiff')

        # Create new folder
        os.makedirs(new_folder, exist_ok=True)

        # Check if destination folder already has files
        if len(os.listdir(new_folder)) > 0:
            print(f"Folder {new_folder} already has files — skipping conversion.")
        else:
            # Convert all .czi files
            for file in tqdm.tqdm(os.listdir(src_folder), desc="Converting CZI to TIFF"):
                if file.endswith('.czi'):
                    czi_path = os.path.join(src_folder, file)
                    out_path = os.path.join(new_folder, file.replace('.czi', '.tiff'))

                    with czifile.CziFile(czi_path) as czi:
                        img = czi.asarray()
                        tifffile.imwrite(out_path, img)
            print("Conversion complete!")

    def _read_image(self):
        """Read in the image to be processed based on its experiment id and return the image and its data."""
        # Find .tiff files
        image_files = [f for f in os.listdir(self.image_dir) if self.experiment in f and f.endswith('.tiff')]
        # Raise error if the experiment file is not unique
        if len(image_files) > 1:
            raise ValueError(f'Expected only one image for experiment {self.experiment}, but found {len(image_files)}')

        # Read in file and associated data
        image = tifffile.TiffFile(f'{self.image_dir}/{image_files[0]}')
        image_data = np.squeeze(image.asarray())
        image_shape = image_data.shape
        
        return image, image_data, image_shape
    
    def _prepare_plotting(self):
        """Prepare the imaging data for plotting."""
        channel_red     = self.image_data[0] 
        channel_green   = self.image_data[1] 

        # Normalize for visualization 
        channel_red_norm    = channel_red / channel_red.max()
        channel_green_norm  = channel_green / channel_green.max()

        shape = channel_red_norm.shape # Adaptive - some images don't have the same shape 

        # Create RG(B) stack for each slice
        rgb_stack = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float32)
        rgb_stack[..., 0] = channel_red_norm
        rgb_stack[..., 1] = channel_green_norm

        return rgb_stack

    def visualize_frame(self, ax=None):
        """Visualize a specific frame of the image."""
        frame = self.rgb_stack[self.frame, :, :, self.color_channel]

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(frame, cmap=self.cmap)
        ax.set_title(f"Frame {self.frame_idx} | {self.cmap}")
        ax.axis('off')

        plt.show()

    def show_annotated_cut(self, save_figure: bool = False):
        # Get points for experiment
        cell_id_to_keep = self.experiment

        # Load the points data
        points_file_loc = os.path.join(self.data_dir, "ablation-lineage/", f"{cell_id_to_keep}.lineage")
        
        # Check that file exists
        if len(os.listdir(points_file_loc)) > 0:
            raise FileNotFoundError(f"The file {points_file_loc} does not exist.")
        # If file exists, extract cut coordinates
        else:
            df = pd.read_csv(points_file_loc, sep='\t')
            # Extract cuts
            df_cuts = df[df.iloc[:, 0].str.contains("cut", case=False, na=False)]

        # Draw the cut
        # Extract the points for the cuts (first two rows in this case)
        x1, y1 = df_cuts.iloc[0]['x'], df.iloc[0]['y']
        x2, y2 = df_cuts.iloc[1]['x'], df.iloc[1]['y']

        img = self.visualize_frame(self.rgb_stack, self.color_channel, cmap=self.cmap)
        img.plot([x1, x2], [y1, y2], color='red', linewidth=2)

        if save_figure:
            path = os.path.join(self.output_path, f"annotated_cut_{self.experiment}.jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        

    def blur_image(self):
        pass

    def find_cell_and_cut_edges(self):
        pass

    def draw_lines(self):
        pass

    def find_cut(self):
        pass

if __name__ == '__main__':

    cut = FindCut
    experiment = 'e_name'
    image_dir = 'id_name'
    output_path = 'output_path_name'
    point_file = 'p_filename'
    interpolation_method = 'method_name'