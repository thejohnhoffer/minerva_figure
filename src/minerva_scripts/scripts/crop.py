""" Test to crop all tiles in a region
"""
from ..load import omero
from minerva_lib.crop import get_lod
from minerva_lib.crop import apply_lod
from minerva_lib.crop import select_tiles
from minerva_lib.crop import stitch_channels
from minerva_lib.crop import get_out_bounds
from minerva_lib.crop import get_tile_bounds
from ..helper import config
from ..helper import api
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
    """ Crop tiles in a region
    """
    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description="combine channels for all tiles"
    )
    cmd.add_argument(
        'url', nargs='?', default='',
        help='OMERO.figure render_scaled_region url'
    )
    cmd.add_argument(
        '-y', default=None,
        help='render_scaled_region: URL: {}'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help="output directory"
    )

    parsed = cmd.parse_args(args)

    # Config if no command line arguments
    key = 'render_scaled_region'
    data = config.load_yaml(parsed.y, key)

    URL = parsed.url
    if not URL:
        default = '/render_scaled_region/548111/0/0/'
        default += '?c=1|0:65535$FF0000,3|0:65535$0000FF'
        default += '&region=-100,-100,1300,1300'
        URL = data.get('URL', default)

    # Read parameters from url in request
    terms = api.scaled_region(URL)

    # Full path format of output files
    out_file = str(pathlib.Path(parsed.o, 'out.png'))

    # Parameters from config url
    all_ranges = terms['r']
    all_colors = terms['c']
    channel_order = terms['chan']
    max_size = terms['max_size']
    k_w, k_h = terms['shape']
    image_id = terms['iid']
    k_time = terms['t']
    k_z = terms['z']
    # Parameters from API request
    n_levels = terms['indices'][2]
    tile_shape = terms['tile']
    px_limit = terms['limit']

    # Compute parameters following figure API
    k_detail = get_lod(n_levels, max_size, k_w, k_h)
    k_origin = apply_lod(terms['origin'], k_detail)
    k_shape = apply_lod(terms['shape'], k_detail)

    # Create variables for cropping
    out = (221 / 255) * np.ones(tuple(k_shape) + (3,))
    args = tile_shape, k_origin, k_shape

    # stitch tiles for all tile indices
    for indices in select_tiles(*args):

        x, y = indices

        # Skip if indices below zero
        if x < 0 or y < 0:
            continue

        # from disk, load all channels for tile
        all_buffer = omero.tile(k_time, k_detail, k_z, y, x,
                                channel_order, image_id, px_limit)
        all_in = zip(all_buffer, all_colors, all_ranges)
        channels = [c for c in map(format_input, all_in) if c]

        # Skip if no channels
        if not channels:
            continue

        # Calculate bounds for given indices
        tile_bounds = get_tile_bounds(indices, *args)
        out_bounds = get_out_bounds(indices, *args)
        stitch_channels(out, tile_bounds, out_bounds, channels)

    # Write the image buffer to a file
    try:
        cv2.imwrite(out_file, 255 * out)
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main(sys.argv)
