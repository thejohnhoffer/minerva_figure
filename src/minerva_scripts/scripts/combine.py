""" Test to combine all channels for all tiles
"""
from ..load import disk
from ..blend import mem
from ..helper import config
import numpy as np
import datetime
import argparse
import pathlib
import cv2
import sys
import os


def parse_config(**kwargs):
    """
    main:
        IN: {DIR:*, NAME:*}
        OUT: {DIR:*, NAME:*, NOW*}
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
    cfg_data = config.load_yaml(kwargs['config'])
    if cfg_data is None:
        # Allow empty config file
        cfg_data = {}
    terms = {}

    # Read root values from config
    in_args = cfg_data.get('IN', {})
    out_args = cfg_data.get('OUT', {})
    terms['t'] = int(cfg_data.get('TIME', 0))
    terms['l'] = int(cfg_data.get('LOD', 0))

    # Validate the threshholds and colors
    terms['r'] = np.float32(cfg_data.get('RANGES', [[0, 1]]))
    terms['c'] = np.float32(cfg_data.get('COLORS', [[1, 1, 1]]))

    # Read the paths with defaults
    in_dir = kwargs.get('i', in_args.get('DIR', '~/tmp/minerva_scripts/in'))
    out_dir = kwargs.get('o', out_args.get('DIR', '~/tmp/minerva_scripts/out'))
    in_name = in_args.get('NAME', '{}_{}_{}_{}_{}_{}.png')
    out_name = out_args.get('NAME', '{}_{}_{}_{}_{}.png')
    # Output stored to current date and time
    now_date = datetime.datetime.now()
    now_time = now_date.time()
    default_date = "{0:04d}_{1:02d}_{2:02d}{4}{3:02d}".format(*[
        now_date.year,
        now_date.month,
        now_date.day,
        now_time.hour,
        os.sep,
    ])
    out_date = out_args.get('NOW', default_date)

    # Format the full paths properly
    out_dir = out_dir.format(NOW=out_date)
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
        help='main: IN: {DIR:*, NAME:*}'
        'OUT: {DIR:*, NAME:*, NOW* TIME: * LOD: *'
        'RANGES: [[*, *]..] COLORS: [[*, *]..]'
    )
    cmd.add_argument(
        '-o', default=argparse.SUPPRESS,
        help="output directory"
    )
    cmd.add_argument(
        '-i', default=argparse.SUPPRESS,
        help="input directory"
    )

    # Actually parse and read arguments
    parsed = vars(cmd.parse_args(args))
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
