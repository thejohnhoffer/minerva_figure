import numpy as np
import glob, re
import cv2
import os

# Load a single image file
def load_file(fmt, args):
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

# Find all the file paths in a range
def find_files(fmt):
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

def blend_tile(t, l, z, y, x, **kwargs):
    """
    Arguments:
        t: integer time step
        l: interger level of detail (powers of 2)
        z: tile offset in depth
        y: vertical tile offset
        x: horizontal tile offset

    Keywords:
        shape: uint16 2vec y,x pixel shape of one tile
        ranges: float32 2 by N-channel min,max range
        colors: float32 2 by N-channel b,g,r max color

    Returns:
        uint8 y by x by 3 color BGR image
    """
    all_bgr = kwargs.get('colors', np.float32([(1.0, 1.0, 1.0)]))
    all_thresh = kwargs.get('ranges', np.float32([(0.0, 1.0)]))
    tile_shape = kwargs.get('shape', np.uint16([1024, 1024]))
    color_tile_shape = tuple(tile_shape) + (3,)

    # numpy buffer for interpolation
    img_buffer = np.zeros(color_tile_shape, dtype=np.float32)

    # Process all channels
    for c in range(n_channel):
        # Fraction of full color
        channel_bgr = all_bgr[c % len(all_bgr)]
        max_color = channel_bgr / float(n_channel)
        # Fraction of full range
        channel_thresh = all_thresh[c % len(all_bgr)]
        lowest, highest = channel_thresh
        clip_size = highest - lowest
        # Load the image file
        params = c, TIME, LOD, z, y, x
        img_data = load_file(in_full_format, params)
        img_data = np.float32(img_data) / 255.0
        # Clip the image
        clip_data = np.clip(img_data, *channel_thresh)
        norm_data = (clip_data - lowest) / clip_size
        # Add the colored data to the image
        y_shape, x_shape = norm_data.shape
        gray_bgr_data = cv2.cvtColor(norm_data, cv2.COLOR_GRAY2RGB)
        img_buffer[0:y_shape, 0:x_shape] += gray_bgr_data * channel_bgr

    return np.uint8(img_buffer*255)

if __name__ == "__main__":

    # Constants
    IN_DIR = "/media/john/420D-AC8E/cycif_images/40BP_59/tiles/"
    IN_NAME_FORMAT = "C{0:}-T{1:}-Z{3:}-L{2:}-Y{4:}-X{5:}.png"

    # Test parameters
    NOW = "2018_04_09/5PM"
    OUT_DIR = "/home/john/2018/data/cycif_out/{}/40BP_59/tiles/".format(NOW)
    OUT_NAME_FORMAT = "T{0:}-Z{2:}-L{1:}-Y{3:}-X{4:}.png"

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Full path format of input files
    in_full_format = os.path.join(IN_DIR, IN_NAME_FORMAT)
    out_full_format = os.path.join(OUT_DIR, OUT_NAME_FORMAT)

    # Find range of image tiles
    ctlzyx_shape, tile_shape = find_files(in_full_format)
    zyx_shape = ctlzyx_shape[-3::]
    n_channel = ctlzyx_shape[0]

    ALL_THRESH = np.float32([
        [0, 1.0],
        [0, 0.1],
        [0, 0.05],
        [0, 0.1],
    ])
    ALL_BGR = np.float32([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ])
    TIME = 0
    LOD = 0
    # Process all z,y,x tiles
    for i in range(np.prod(zyx_shape)):
        z,y,x = np.unravel_index(i, zyx_shape)

        # DERP
        if z != 0:
            continue

        img_buffer = blend_tile(TIME, LOD, z, y, x, **{
            'colors': ALL_BGR,
            'ranges': ALL_THRESH,
            'shape': tile_shape
        })

        # Write the image buffer to a file
        out_file = out_full_format.format(TIME, LOD, z, y, x)
        try:
            cv2.imwrite(out_file, img_buffer)
        except Exception as e:
            print (e)

