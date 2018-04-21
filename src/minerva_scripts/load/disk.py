import numpy as np
import glob
import re
import cv2
import os


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
    # Print error if nothing loaded
    msg = "no channel #{0}: t{1}, l{2}, z{3}, y{4}, x{5}"
    print(msg.format(*args))
    return None


def tile(t, l, z, y, x, n_channel=1, in_fmt=None):
    """Load all channels for a given tile
    Arguments:
        t: integer time step
        l: interger level of detail (powers of 2)
        z: tile offset in depth
        y: vertical tile offset
        x: horizontal tile offset
        n_channel: total number of channels to load
        in_fmt: string defining file pattern

    Returns:
        list of numpy image channels for a tile
    """
    if in_fmt is None:
        in_fmt = '{}_{}_{}_{}_{}_{}.png'

    # Load all channels
    const = t, l, z, y, x
    c_range = range(n_channel)
    return [image(in_fmt, c, *const) for c in c_range]


def format_input(args):
    ''' Combine all parameters
    '''
    image_, color_, range_ = args
    return {
        'image': image_,
        'color': color_,
        'min': range_[0],
        'max': range_[1],
    }


def index(fmt):
    """Find all the file paths in a range

    Args:
        fmt: string defining file pattern

    Returns:
        size in channels, times, LOD, Z, Y, X
        image tile size in pixels: y, x
    """
    num_dim = 6
    pixels = np.uint16([0, 0])
    dims = range(1, num_dim + 1)
    sizes = np.zeros(num_dim, dtype=np.uint16)

    # Interpret the format string
    fmt_order = fmt.format(*dims)
    fmt_iglob = fmt.format(*(('*',) * num_dim))
    fmt_regex = fmt.format(*(('(\d+)',) * num_dim))

    # Get the order of the parameters
    re_order = re.match(fmt_regex, fmt_order)
    order = list(map(int, map(re_order.group, dims)))

    # Find all files matching the pattern
    for name in glob.iglob(fmt_iglob):
        # Extract parameters for each dimension
        match = next(re.finditer(fmt_regex, name), None)
        if match is not None:
            coords = list(map(match.group, order))
            # Take the maximum of all coordinates
            sizes = np.maximum(sizes, 1 + np.uint16(coords))
            # Read first image
            if not all(pixels):
                file_name = match.group(0)
                file_data = cv2.imread(file_name, 0)
                pixels[:] = file_data.shape

    return sizes, pixels
