""" Test to crop all tiles in a region
"""
from ..load import disk
from memory_profiler import profile
from minerva_lib.blend import composite_channels
from ..helper import config
import argparse
import pathlib
import sys


@profile
def debug_crop(all_in):
    ''' Combine all inputs
    '''
    pass


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
    # in_path_format = terms['i']
    # out_path_format = terms['o']

    print(terms)


if __name__ == "__main__":
    main(sys.argv)
