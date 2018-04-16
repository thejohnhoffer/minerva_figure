""" Test to combine all channels for all tiles
"""
from ..load import disk
from ..blend import mem
from ..helper import config
import numpy as np
import argparse
import pathlib
import cv2
import sys
import os


def parse_config(**kwargs):
    """
    main:
        RANGES: [[*, *]..]
        COLORS: [[*, *]..]
        TIME: *
        LOD: *

    Keyword Arguments:
        config: path to yaml with above keys
        o: output directory
        i: input directory

    Returns:
        t: integer timestep
        l: integer power-of-2 level-of-detail
        r: float32 N channels by 2 min, max
        c: float32 N channels by 3 b, g, r
        o: full output format
        i: full input format
    """
    in_name = 'C{0:}-T{1:}-Z{3:}-L{2:}-Y{4:}-X{5:}.png'
    out_name = 'T{0:}-Z{2:}-L{1:}-Y{3:}-X{4:}.png'
    cfg_data = config.load_yaml(kwargs['config'])
    # Allow empty config file
    if cfg_data is None:
        cfg_data = {}
    terms = {}

    # Read root values from config
    terms['t'] = int(cfg_data.get('TIME', 0))
    terms['l'] = int(cfg_data.get('LOD', 0))

    # Validate the threshholds and colors
    terms['r'] = np.float32(cfg_data.get('RANGES', [[0, 1]]))
    terms['c'] = np.float32(cfg_data.get('COLORS', [[1, 1, 1]]))

    # Read the paths with defaults
    try:
        in_dir = kwargs['o']
        out_dir = kwargs['i']
    except KeyError as k_e:
        raise k_e

    # Join the full paths properly
    terms['i'] = str(pathlib.Path(in_dir, in_name))
    terms['o'] = str(pathlib.Path(out_dir, out_name))

    # Create output directory if nonexistant
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return terms


def main(args=sys.argv[1:]):
    """ Combine channels for all tiles
    """
    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description="combine channels for all tiles"
    )
    cmd.add_argument(
        'config', nargs='?', default='config.yaml',
        help='main: {'
        ' TIME: * LOD: *'
        ' RANGES: [[*, *]..]'
        ' COLORS: [[*, *]..]'
        ' }'
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
    terms = parse_config(**parsed)

    # Full path format of input files
    in_path_format = terms['i']
    out_path_format = terms['o']
    # Important parameters
    all_ranges = terms['r']
    all_colors = terms['c']
    k_time = terms['t']
    k_detail = terms['l']

    # Find range of image tiles
    ctlzyx_shape, tile_shape = disk.index(in_path_format)
    zyx_shape = ctlzyx_shape[-3::]
    n_channel = ctlzyx_shape[0]

    # Process all z, y, x tiles
    for i in range(np.prod(zyx_shape)):
        z, y, x = np.unravel_index(i, zyx_shape)

        # DERP
        if z != 0:
            continue

        # from disk, load all channels for tile
        all_buffer = disk.tile(k_time, k_detail, z, y, x, **{
            'format': in_path_format,
            'count': n_channel,
        })

        # Continue if no channel buffers for given tile
        all_buffer = [b for b in all_buffer if b is not None]
        if not all_buffer:
            continue

        # from memory, blend all channels loaded
        img_buffer = mem.tile(all_buffer, **{
            'ranges': all_ranges,
            'shape': tile_shape,
            'colors': all_colors,
        })

        # Write the image buffer to a file
        out_file = out_path_format.format(k_time, k_detail, z, y, x)
        try:
            cv2.imwrite(out_file, img_buffer)
        except OSError as o_e:
            print(o_e)


if __name__ == "__main__":
    main(sys.argv)
