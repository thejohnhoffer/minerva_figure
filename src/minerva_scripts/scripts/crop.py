""" Test to crop all tiles in a region
"""
from ..load import disk
from minerva_lib.crop import get_lod
from minerva_lib.crop import apply_lod
from minerva_lib.crop import select_tiles
from minerva_lib.crop import stitch_channels
from minerva_lib.crop import get_out_bounds
from minerva_lib.crop import get_tile_bounds
from ..helper import config
import numpy as np
import argparse
import pathlib
import cv2
import sys


def format_input(args):
    ''' Combine all parameters
    '''
    image_, color_, range_ = args
    if image_ is None:
        return None
    return {
        'image': image_,
        'color': color_,
        'min': range_[0],
        'max': range_[1],
    }


def main(args=sys.argv[1:]):
    """ Combine tiles in a region
    """
    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description="combine channels for all tiles"
    )
    cmd.add_argument(
        'config', nargs='?', default='config.yaml',
        help='See config.parse for behavior of the keys:'
        ' main, render_scaled_region'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help="output directory"
    )
    cmd.add_argument(
        '-i', required="True",
        help="input directory"
    )

    parsed = vars(cmd.parse_args(args))
    # Actually parse and read arguments
    terms = config.parse('region', **parsed)

    # Full path format of input files
    in_path_format = terms['i']
    out_path_format = terms['o']

    # Parameters from config url
    all_ranges = terms['r']
    all_colors = terms['c']
    channel_order = terms['chan']
    max_size = terms['max_size']
    k_w, k_h = terms['shape']
    k_time = terms['t']
    k_z = terms['z']

    # Parameters from file system
    extents, tile_shape = disk.index(in_path_format)
    n_levels = extents[2]

    # Compute parameters following figure API
    k_detail = get_lod(n_levels, max_size, k_w, k_h)
    k_origin = apply_lod(terms['origin'], k_detail)
    k_shape = apply_lod(terms['shape'], k_detail)

    # Create variables for cropping
    out = np.zeros(tuple(k_shape) + (3,))
    args = tile_shape, k_origin, k_shape

    # stitch tiles for all tile indices
    for indices in select_tiles(*args):

        x, y = indices

        # Skip if indices below zero
        if x < 0 or y < 0:
            continue

        # from disk, load all channels for tile
        all_buffer = disk.tile(k_time, k_detail, k_z, y, x,
                               channel_order, in_path_format)
        all_in = zip(all_buffer, all_colors, all_ranges)
        channels = [c for c in map(format_input, all_in) if c]

        # Calculate bounds for given indices
        tile_bounds = get_tile_bounds(indices, *args)
        out_bounds = get_out_bounds(indices, *args)
        stitch_channels(out, tile_bounds, out_bounds, channels)

    # Write the image buffer to a file
    out_file = out_path_format.format(k_time, k_detail, k_z, 0, 0)
    try:
        cv2.imwrite(out_file, 255 * out)
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main(sys.argv)
