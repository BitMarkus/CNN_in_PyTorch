# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
# ===== Third-Party Imports =====
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
import cv2
# ===== Own Modules =====
from settings import setting

# Create all working folders in the root directory of the program if they do not exist.
def create_prg_folders() -> None:
    setting["pth_input"].mkdir(parents=True, exist_ok=True)
    setting["pth_output"].mkdir(parents=True, exist_ok=True)

# Read a czi mosaic file.
# Args:
#   czi_data: CziFile object
#   channel (int): Channel index. Defaults to 0.
#   z_plane (int): Z-plane index. Defaults to 0.
# Returns:
#   numpy.ndarray: Mosaic image data
def read_czi_mosaic(czi_data, channel: int = 0, z_plane: int = 0):
    return czi_data.read_mosaic(C=channel, Z=z_plane, scale_factor=setting['preproc_czi_import_scale'])

# Return a list of all czi files in the input folder without path and extension.
# Returns:
#   list: Sorted list of filenames without extension
def get_czi_file_list() -> list:
    czi_list = list(setting['pth_input'].glob("*" + setting['preproc_czi_img_ext']))
    czi_list = [czi_pth.name.replace(czi_pth.suffix, '') for czi_pth in czi_list]
    czi_list.sort()
    return czi_list

# Convert normalized numpy array (0-1) to uint8 (0-255).
# Args:
#   np_arr (numpy.ndarray): Input array normalized to [0, 1]
# Returns:
#   numpy.ndarray: uint8 array in range [0, 255]
def convert_to_8bit(np_arr) -> np.ndarray:
    np_arr *= 255
    return np_arr.astype(np.uint8)

# Normalize the image based on percentiles to handle outliers.
# Args:
#   image (numpy.ndarray): Input image
# Returns:
#   numpy.ndarray: Normalized image in range [0, 1]
def percentile_normalization(image) -> np.ndarray:
    lower_value = np.percentile(image, setting['preproc_perc_min'])
    upper_value = np.percentile(image, setting['preproc_perc_max'])
    normalized_image = np.clip((image - lower_value) / (upper_value - lower_value), 0, 1)
    return normalized_image

# Reduce noise with Gaussian blur.
# Args:
#   np_arr (numpy.ndarray): Input image
#   kernel_size (tuple): Kernel size. Defaults to (5, 5).
# Returns:
#   numpy.ndarray: Blurred image
def reduce_noise(np_arr, kernel_size: tuple = (5, 5)) -> np.ndarray:
    return cv2.GaussianBlur(np_arr, kernel_size, 0)

# Create folder for each czi image in the output folder.
# Args:
#   file_name (str): Name of the czi file
# Returns:
#   bool: True if successful
def create_export_folder(file_name: str) -> bool:
    output_pth = Path.joinpath(setting['pth_output'], file_name)
    output_pth.mkdir(parents=True, exist_ok=True)
    print(f"Folder {output_pth} for exported images was successfully created.")
    return True

# Read the dimensions (x, y) of a tile in a czi mosaic.
# Args:
#   czi_data: CziFile object
# Returns:
#   dict: Dictionary with 'x' and 'y' keys
def read_tile_size(czi_data) -> dict:
    shape = czi_data.get_dims_shape()
    return {'x': int(shape[0]['X'][1]), 'y': int(shape[0]['Y'][1])}

# Read the number of z-planes in the czi image.
# Args:
#   czi_data: CziFile object
# Returns:
#   int: Number of z-planes
def read_num_z_planes(czi_data) -> int:
    shape = czi_data.get_dims_shape()
    return int(shape[0]['Z'][1])

# Get origin (upper left corner) of a specific tile.
# Args:
#   czi_data: CziFile object
#   tile_index (int): 1D tile index
# Returns:
#   tuple: (x, y) origin coordinates
def get_tile_origin(czi_data, tile_index: int) -> tuple:
    channel = 0
    z_plane = 0
    tile0_data = czi_data.get_mosaic_tile_bounding_box(M=0, C=channel, Z=z_plane)
    tile_data = czi_data.get_mosaic_tile_bounding_box(M=tile_index, C=channel, Z=z_plane)
    return ((tile_data.x - tile0_data.x), (tile_data.y - tile0_data.y))

# Convert 2D mosaic tile index to 1D.
# Args:
#   index_2d_row (int): Row index
#   index_2d_col (int): Column index
#   num_cols (int): Number of columns
# Returns:
#   int: 1D tile index
def convert_index_2d_to_1d(index_2d_row: int, index_2d_col: int, num_cols: int) -> int:
    return index_2d_col + (index_2d_row * num_cols)

# Convert numpy array to PIL image.
# Args:
#   np_array (numpy.ndarray): Input array
#   mode (str): PIL image mode. Defaults to 'L' (grayscale).
# Returns:
#   PIL.Image: PIL image
def pil_from_np(np_array, mode: str = 'L') -> Image.Image:
    return Image.fromarray(np_array, mode=mode)

# Resize PIL image.
# Args:
#   pil_img (PIL.Image): Input PIL image
# Returns:
#   PIL.Image: Resized PIL image
def resize_pil_img(pil_img: Image.Image) -> Image.Image:
    return pil_img.resize((setting['preproc_slice_resize']['x'], setting['preproc_slice_resize']['y']))

# Calculate the slice origin for the 1x slice in each tile.
# Args:
#   czi_data: CziFile object
#   tile_pos_x (int): Tile X position
#   tile_pos_y (int): Tile Y position
#   num_tiles_x (int): Number of tiles in X direction
# Returns:
#   dict: Dictionary with 'x' and 'y' slice origins
def get_slice_origin(czi_data, tile_pos_x: int, tile_pos_y: int, num_tiles_x: int) -> dict:
    tile_size = read_tile_size(czi_data)
    tile_index = convert_index_2d_to_1d(tile_pos_x, tile_pos_y, num_tiles_x)
    tile_origin_x, tile_origin_y = get_tile_origin(czi_data, tile_index)
    x_offset = (tile_size['x'] - (setting['preproc_slice_size']['x'])) // 2
    y_offset = (tile_size['y'] - (setting['preproc_slice_size']['y'])) // 2
    slice_origin_x = tile_origin_x + x_offset
    slice_origin_y = tile_origin_y + y_offset
    return {'x': slice_origin_x, 'y': slice_origin_y}

# Slice numpy image.
# Args:
#   np_arr (numpy.ndarray): Input image
#   origin (dict): Origin coordinates with 'x' and 'y' keys
# Returns:
#   numpy.ndarray: Sliced image
def slice_np_img(np_arr, origin: dict) -> np.ndarray:
    slice_end = {'x': (origin['x'] + setting['preproc_slice_size']['x']), 'y': (origin['y'] + setting['preproc_slice_size']['y'])}
    return np_arr[origin['y']:slice_end['y'], origin['x']:slice_end['x']]

#######################
# Find sharpest image #
#######################

# Tenengrad sharpness metric. Higher values indicate sharper images.
# Args:
#   image (numpy.ndarray): Input grayscale image
# Returns:
#   float: Sharpness score
def tenengrad_sharpness(image: np.ndarray) -> float:
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    return np.sum(grad_x**2 + grad_y**2)

# Find the sharpest plane in a z-stack using a given focus metric.
# Args:
#   z_stack (list): List of numpy arrays (z-planes)
#   metric (callable): Focus metric function
# Returns:
#   tuple: (sharpest_image, sharpest_index)
def find_sharpest_plane(z_stack: list, metric) -> tuple:
    focus_values = [metric(plane) for plane in z_stack]
    sharpest_plane_index = np.argmax(focus_values)
    return z_stack[sharpest_plane_index], sharpest_plane_index