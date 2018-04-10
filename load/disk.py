import numpy as np
import glob, re
import cv2
import os

# Load a single image file
def image(fmt, *args):
    """ Load a single file by pattern

    Args:
        fmt: string defining file pattern
        args: tuple completing pattern

    Returns:
        numpy array loaded from file
    """
    img_name = fmt.format(*args)
    if os.path.exists(img_name):
        return cv2.imread(img_name, 0)
    return None

# Load all channels for a given tile
def tile(t, l, z, y, x, **kwargs):
    """
    Arguments:
        t: integer time step
        l: interger level of detail (powers of 2)
        z: tile offset in depth
        y: vertical tile offset
        x: horizontal tile offset

    Keywords:
        shape: uint16 2vec y,x pixel shape of one tile
        count: total number of channels to load
        format: string defining file pattern

    Returns:
        list of numpy image channels for a tile
    """
    in_fmt = kwargs.get('format', '{}_{}_{}_{}_{}_{}.png')
    n_channel = kwargs.get('count', 1)

    # Load all channels
    const = t, l, z, y, x
    c_range = range(n_channel)
    return [image(in_fmt, c, *const) for c in c_range]

# blend all channels given
def blend(all_buffer, **kwargs):
    """
    Arguments:
        all_buffer: list of numpy image channels for a tile

    Keywords:
        ranges: float32 2 by N-channel min,max range
        colors: float32 2 by N-channel b,g,r max color

    Returns:
        uint8 y by x by 3 color BGR image
    """
    all_bgr = kwargs.get('colors', np.float32([(1.0, 1.0, 1.0)]))
    all_thresh = kwargs.get('ranges', np.float32([(0.0, 1.0)]))
    tile_shape = kwargs.get('shape', np.uint16([1024, 1024]))

    # final buffer for blending
    color_tile_shape = tuple(tile_shape) + (3,)
    img_buffer = np.zeros(color_tile_shape, dtype=np.float32)

    # Process all channels
    for c_bgr, c_thresh, c_img in zip(all_bgr, all_thresh, all_buffer):
        # Scale the dynamic range to 0.0 - 1.0
        n_bits = 8 * c_img.itemsize
        bit_factor = 1.0 / (2.0 ** n_bits)
        img_data = np.float32(c_img * bit_factor)
        # Maximum color for this channel
        avg_factor = 1.0 / len(all_buffer)
        color_factor = 256.0 * c_bgr * avg_factor
        # Fraction of full range
        lowest, highest = c_thresh
        clip_size = highest - lowest
        # Clip the image
        clip_data = np.clip(img_data, lowest, highest)
        norm_data = (clip_data - lowest) / clip_size
        # Add the colored data to the image
        y_shape, x_shape = norm_data.shape
        gray_bgr_data = cv2.cvtColor(norm_data, cv2.COLOR_GRAY2RGB)
        img_buffer[0:y_shape, 0:x_shape] += gray_bgr_data * color_factor

    return np.uint8(img_buffer)

# Find all the file paths in a range
def index(fmt):
    """ Parse directory structure

    Args:
        fmt: string defining file pattern

    Returns:
        size in channels, times, LOD, Z, Y, X
        image tile size in pixels: y, x
    """
    num_dim = 6
    voxels = np.uint16([1, 0, 0])
    sizes = np.zeros(num_dim, dtype=np.uint16)
    all_iglob = fmt.format(*(('*',) * num_dim))
    all_regex = fmt.format(*(('(\d+)',) * num_dim))

    # Find all files matching the pattern
    for name in glob.iglob(all_iglob):
        # Extract parameters for each dimension
        match = next(re.finditer(all_regex, name), None)
        if match is not None:
            coords = match.group(*range(1,num_dim+1))
            # Take the maximum of all coordinates
            sizes = np.maximum(sizes, np.uint16(coords))
            # Read first image
            if not all(voxels):
                file_name = match.group(0)
                file_data = cv2.imread(file_name, 0)
                voxels[-2::] = file_data.shape

    # zero-based sizes
    return sizes + 1, voxels[-2::]
