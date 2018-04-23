""" Test to combine all channels for all tiles
"""
from ..load import disk
from memory_profiler import profile
from minerva_lib.blend import linear_bgr
from ..helper import config
import numpy as np
import argparse
import pathlib
import cv2
import sys


@profile
def debug_linear_bgr(all_in):
    ''' Combine all inputs
    '''
    return linear_bgr(list(map(disk.format_input, all_in)))


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
    terms = config.parse(**parsed)

    # Full path format of input files
    in_path_format = terms['i']
    out_path_format = terms['o']

    # Important parameters
    channel_order = terms['chan']
    all_ranges = terms['r']
    all_colors = terms['c']
    k_time = terms['t']
    k_detail = terms['l']

    # Find range of image tiles
    ctlzyx_shape, tile_shape = disk.index(in_path_format)
    zyx_shape = ctlzyx_shape[-3::]

    # Process all z, y, x tiles
    for i in range(np.prod(zyx_shape)):
        z, y, x = np.unravel_index(i, zyx_shape)

        # from disk, load all channels for tile
        all_buffer = disk.tile(k_time, k_detail, z, y, x,
                               channel_order, in_path_format)

        # Continue if no channel buffers for given tile
        all_buffer = [b for b in all_buffer if b is not None]
        if not all_buffer:
            continue

        # from memory, blend all channels loaded
        all_in = zip(all_buffer, all_colors, all_ranges)

        img_buffer = debug_linear_bgr(all_in)
        # Write the image buffer to a file
        out_file = out_path_format.format(k_time, k_detail, z, y, x)
        try:
            cv2.imwrite(out_file, img_buffer)
        except OSError as o_e:
            print(o_e)


if __name__ == "__main__":
    main(sys.argv)
