import numpy as np
import glob
import re
import os

import skimage.io
from minerva_lib import blend

#####
# Load from disk
###


class disk():

    @staticmethod
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
            return skimage.io.imread(img_name)

        return None

    @staticmethod
    def tile(t, l, z, y, x, c_order, in_fmt=None):
        """Load all channels for a given tile
        Arguments:
            t: integer time step
            l: interger level of detail (powers of 2)
            z: tile offset in depth
            y: vertical tile offset
            x: horizontal tile offset
            c_order: list of channels to load
            in_fmt: string defining file pattern

        Returns:
            list of numpy image channels for a tile
        """
        if in_fmt is None:
            in_fmt = '{}_{}_{}_{}_{}_{}.png'

        # Load all channels
        const = t, l, z, y, x
        return [disk.image(in_fmt, c, *const) for c in c_order]

    @staticmethod
    def index(fmt):
        """Find all the file paths in a range

        Args:
            fmt: string defining file pattern

        Returns:
            size in channels, times, LOD, Z, Y, X
            image tile size in pixels: y, x
        """
        num_dim = 6
        pixels = np.array([0, 0], dtype=np.uint16)
        dims = range(1, num_dim + 1)
        sizes = np.zeros(num_dim, dtype=np.uint16)

        # Interpret the format string
        fmt_order = fmt.format(*dims)
        fmt_iglob = fmt.format(*(('*',) * num_dim))
        fmt_regex = fmt.format(*((r'(\d+)',) * num_dim))

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
                    file_data = skimage.io.imread(file_name)
                    pixels[:] = file_data.shape

        return sizes, pixels
######
# Entrypoint
###


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


def main():
    """ Crop a region
    """

    out = './output'
    os.makedirs(out)
    out_path_format = out + '/T{0:}-Z{2:}-L{1:}-Y{3:}-X{4:}.png'

    # Important parameters
    channels = [0, 1]
    ranges = np.array([
      [0, 1],
      [0.006, 0.024]
    ])
    colors = np.array([
      [1, 0, 0],
      [0, 1, 0]
    ])
    x = 0
    y = 0
    z = 0
    t = 0
    level = 0

    # Full path format of input files
    in_path_format = '/home/j/data/2018/06/png_tiles/'
    in_path_format += 'C{0:}-T{1:}-Z{3:}-L{2:}-Y{4:}-X{5:}.png'

    # from disk, load all channels for tile
    images = disk.tile(t, level, z, y, x,
                       channels, in_path_format)

    all_in = zip(images, colors, ranges)
    inputs = list(map(format_input, all_in))
    img_buffer = 255*blend.composite_channels(inputs)

    # Write the image buffer to a file
    out_file = out_path_format.format(t, level, z, y, x)
    try:
        skimage.io.imsave(out_file, np.uint8(img_buffer))
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main()
