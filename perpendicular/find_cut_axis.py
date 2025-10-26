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
    """A class for finding the cut caused by Cortical Laser Ablation in a microscopy video."""

    def __init__(self, experiment: str, data_dir: str = '../data', output_path: str = 'perpendicular/out/', frame_idx: int = 4, color_channel: int = 0, cmap: str = 'gray', interpolation_method: str = cv2.INTER_CUBIC):
        self.experiment = experiment
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

        if not self.cmap in ['gray', 'viridis', 'magma']:
            raise ValueError('Cmap has to be one of the following: gray, viridis, magma.')
        
        if not self.color_channel in [0, 1]:
            raise ValueError('Color channel has to be either 0 (red), or 1 (green).')
        
        # Convert images
        self._convert_images_to_tiff()
        self.image_dir = os.path.join(os.path.join(self.data_dir, 'tiff'))

        # Read in image
        self.tiff_image, self.image_data, self.image_shape = self._read_image()

        # Prepare for plotting
        self.rgb_stack = self._prepare_plotting()

        # Image used for plotting
        self.image = self.rgb_stack[self.frame_idx, :, :, self.color_channel]

        # Initiate "original image has been updated" to False, and modifications done as an empty list
        self.origin_updated = False
        self.modifications = []

    def __str__(self) -> str:
        pass

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
        if not len(os.listdir(new_folder)) > 0:
            # Convert all .czi files
            for file in tqdm.tqdm(os.listdir(src_folder), desc="Converting CZI to TIFF"):
                if file.endswith('.czi'):
                    czi_path = os.path.join(src_folder, file)
                    out_path = os.path.join(new_folder, file.replace('.czi', '.tiff'))

                    with czifile.CziFile(czi_path) as czi:
                        img = czi.asarray()
                        tifffile.imwrite(out_path, img)
            print("Conversion of .czi file to .tiff files complete!")

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

    def visualize_frame(self, ax = None, save_figure: bool = False, show: bool = True):
        """
        Visualize a specific frame of the image. If save_figure is set to true, the image 
        will be saved to the specified output path. If show is set to True (default) the 
        image will be printed to the screen.
        """
 
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.image, cmap=self.cmap)

        # Either save the figure to a path, show it, or pass it forward to other methods to use
        if save_figure:
            ax.set_title(f"Frame {self.frame_idx} | {self.cmap}")
            ax.axis('off')
            path = os.path.join(self.output_path, f"frame_{self.frame_idx}_{self.experiment}.jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Figure was saved to {path}.")
        
        if show:
            ax.set_title(f"Frame {self.frame_idx} | {self.cmap}")
            ax.axis('off')
            plt.show()
        
        return ax  # Return axes for further plotting

    def visualize_annotated_cut(self, save_figure: bool = False):
        """
        Shows the cut on top of the frame according to the annotated metadata related to
        the experiment. If save_figure is set to True, the image will be saved to the
        specified output path.
        """
        
        # Get points for experiment
        cell_id_to_keep = self.experiment

        # Load the points data
        points_file_loc = os.path.join(self.data_dir, "ablation-lineage/", f"{cell_id_to_keep}.lineage")
                
        # Check that file exists
        if not os.path.isfile(points_file_loc):
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

        # Get the image axes from visualize_frame (without showing)
        ax = self.visualize_frame(show=False)

        # Draw the annotated cut
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)

        # Save or show
        if save_figure:
            path = os.path.join(self.output_path, f"annotated_cut_{self.experiment}.jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Figure was savaed to {path}.")
        else:
            ax.set_title('Annotated Cut')
            ax.axis('off')
            plt.show()

    def show_current_image_info(self, show_image: bool = True, print_info: bool = True):
        """Shows the current image, with information on how many modifications have been done."""
        if show_image:
            plt.imshow(self.image, cmap=self.cmap); plt.title(f'Current Image ({len(self.modifications)} modifications applied)'); plt.axis('off')
            plt.show()

        if print_info:
            print(f"The following modifications have been applied:\n{self.modifications}")

    def blur_image(self, blur_kernel: int = 31, sigma: int = 3, update_origin: bool = True, show_blurred: bool = True, save_figure: bool = False, show_versions: bool = False, save_versions: bool = False):
        """
        Blurs the image by first downsampling and then applying a Gaussian Blur. Then it is 
        upsampled again for the next steps of the process. If show_versions is set to True, 
        a comparison plot will be created and shown, portraying the differences between the
        original image, the downsampled one, and the downsampled and blurred image. If wished,
        the figure can be saved to the ouput path.
        """
        # Downsample the image
        downsampled = cv2.resize(self.image, (132, 132), interpolation=self.interpolation_method)
        # Add Gaussian blur
        downsampled_blurred = cv2.GaussianBlur(downsampled, (blur_kernel, blur_kernel), sigma)
        # Upsample to original size 
        reupsampled = cv2.resize(downsampled_blurred, (512, 512), interpolation=self.interpolation_method)

        if show_blurred or save_figure:
            plt.imshow(reupsampled, cmap=self.cmap); plt.title('Blurred Image'); plt.axis('off')
            if save_figure:
                path = os.path.join(self.output_path, f"blurred_{self.experiment}.jpg")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"Blurred image was saved to {path}.")
            if show_blurred:    
                plt.show()

        if show_versions or save_versions:
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(self.image, cmap=self.cmap); plt.title('Original Image'); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(downsampled, cmap=self.cmap); plt.title('Downsampled Image'); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(downsampled_blurred, cmap=self.cmap); plt.title('Blurred Image'); plt.axis('off')
            plt.tight_layout()
            if save_versions:
                path = os.path.join(self.output_path, f"comparison_blurred_{self.experiment}.jpg")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"The blurred image with comparisons to original and downsampled-only was saved to {path}.")
            if show_versions:
                plt.show()
            
        # Updates the original image as part of the FindCut-algorithm
        if update_origin:
            self.image = reupsampled
            self.modifications.append('Gaussian Blur')
            self.origin_updated = True
            print("Original picture has now been updated as a blurred version!")
            return self

    def find_cell_and_cut_edges(self):
        pass

    def draw_lines(self):
        pass

    def find_cut(self):
        pass

if __name__ == '__main__':

    """
    OBSERVE!
    I have declared the data path in the class initiation according to where it is on my computer.
    Make sure to change it to match where your data lies.
    """
    experiment = '220530-E1_Out'
        # experiment = '220726-E6_Out'
        # experiment = '220914-E4_Out'
        # experiment = '220914-E2_Out'
        # experiment = '220627-E3_Out'

    cut = FindCut(experiment)
    # cut.visualize_frame()
    # cut.visualize_annotated_cut()
    cut.blur_image(show_blurred=False)
    cut.show_current_image_info()
