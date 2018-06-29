""" Test to crop all tiles in a region
"""
from ..load import omero
from minerva_lib import crop
from ..helper import config
from ..helper import api
import numpy as np
import skimage.io
import argparse
import pathlib
import sys


def format_input(args):
    ''' Combine all parameters
    '''
    id_, color_, range_ = args
    return {
        'channel': id_,
        'color': color_,
        'min': range_[0],
        'max': range_[1],
    }


def do_crop(load_tile, channels, tile_size, origin, shape,
            levels=1, max_size=2000, order='before'):
    ''' Interface with minerva_lib.crop

    Args:
        load_tile: Function to supply 2D numpy array
        channels: List of dicts of channel rendering settings
        tile_size: The width, height of a single tile
        origin: Request's full-resolution x, y origin
        shape: Request's full-resolution width, height
        levels: The number of levels of detail
        max_size: The maximum response width or height
        order: Composite 'before' or 'after' stitching

    Returns:
        2D numpy float array of with width, height given by
        `shape` if `shape <= max_size` or width, height given by
        `shape / 2 ** l <= max_size` for the lowest `l` meeting
        `l < levels`. The array is a composite of all channels
        for full or partial tiles within `shape` from `origin`.
    '''

    # Compute parameters following figure API
    k_lod = crop.get_lod(levels, max_size, *shape)
    k_origin = crop.apply_lod(origin, k_lod)
    k_shape = crop.apply_lod(shape, k_lod)

    msg = '''Cropping 1/{0} scale:
    {2} pixels starting at {1}
    '''.format(2**k_lod, k_origin, k_shape)
    print(msg)

    # Minerva reads tile
    def store_tile(tile):
        c = tile['channel']
        i, j = tile['indices']

        # Disallow negative tiles
        if i < 0 or j < 0:
            return None

        image = load_tile(c, k_lod, i, j)

        # Disallow empty images
        if image is None:
            return None

        tile['image'] = image
        return tile

    # Load and stitch all tiles
    tiles = crop.iterate_tiles(channels, tile_size,
                               k_origin, k_shape)
    return crop.stitch_tiles(map(store_tile, tiles),
                             tile_size, k_shape, order)


def main(args):
    """ Crop a region
    """
    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description="Crop a region"
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
        '--after', action='store_true',
        help='composite after stitching'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help='output directory'
    )

    parsed = cmd.parse_args(args)
    out_file = str(pathlib.Path(parsed.o, 'out.png'))
    order = ['before', 'after'][parsed.after]

    # Config if no command line arguments
    default = '/548111/0/0/?c=1|0:65535$FF0000,3|0:65535$0000FF'
    default += '&region=-100,-100,1300,1300'
    data = config.load_yaml(parsed.y, 'render_scaled_region')
    URL = parsed.url if parsed.url else data.get('URL', default)

    # Read parameters from URL and API
    keys = api.scaled_region(URL)

    # Make array of channel parameters
    inputs = zip(keys['chan'], keys['c'], keys['r'])
    channels = map(format_input, inputs)

    # OMERO loads the tiles
    def ask_omero(c, l, i, j):
        return omero.image(c, keys['limit'], keys['iid'],
                           keys['z'], keys['t'], l, i, j)

    # Minerva does the cropping
    out = do_crop(ask_omero, channels, keys['tile_size'],
                  keys['origin'], keys['shape'], keys['levels'],
                  keys['max_size'], order)

    # Write the image buffer to a file
    try:
        skimage.io.imsave(out_file, np.uint8(255 * out))
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main(sys.argv[1:])
