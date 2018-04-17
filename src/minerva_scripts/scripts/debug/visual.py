""" Compare blend results with expected output
"""
import numpy as np
from ...blend import mem
from .features import BlendTile
from .features import Log


class Visual(BlendTile):
    """ Constants used for testing
    """
    u16 = np.uint16(range(2**16))

    # Sample named colors
    white = np.float32([1, 1, 1])
    yellow = np.float32([0, 1, 1])
    green = np.float32([0, 1, 0])
    blue = np.float32([1, 0, 0])
    red = np.float32([0, 0, 1])

    # Test input images
    u16_all = BlendTile.square(u16)
    u16_7x7 = BlendTile.square(u16[:49])
    # On/off grid
    u16_chess0 = np.uint16(u16_7x7 % 2 == 0)*u16[-1]
    u16_chess1 = np.uint16(u16_7x7 % 2 == 1)*u16[-1]

    @staticmethod
    def sanity_check(*args):
        """ Compare basic details of two images

        Arguments:
            t_pair: list of two images to compare
            t_id: identity of test
        """
        t_pair = args[0]
        type_pair = [x.dtype for x in t_pair]
        shape_pair = [x.shape for x in t_pair]
        # Log messages if results are unexpected
        type_msg = "dtypes differ: truth {}, result {}"
        shape_msg = "shapes differ: truth {}, result {}"

        # Assume same types
        if len(set(type_pair)) != 1:
            err_msg = type_msg.format(*type_pair)
            raise ValueError(err_msg)

        # Assume same shapes
        if np.subtract(*shape_pair).any():
            err_msg = shape_msg.format(*shape_pair)
            raise ValueError(err_msg)

    @staticmethod
    def visual_check(t_pair, t_id='?'):
        """ Visually compare two images

        Arguments:
            t_pair: list of two images to compare
            t_id: id of test for output images
        """
        # Write out actual images
        diff_image = np.subtract(*t_pair)
        Log.write_image(t_pair[0], t_id+'_ok')
        Log.write_image(t_pair[1], t_id+'_out')
        Log.write_image(diff_image, t_id+'_diff')


def generic_test_tile(t_chans, t_id, t_keys, t_ok, t_list=None):
    """ Run test on tile blend function

    Arguments:
        t_id: str name of test
        t_keys: keywords for call
        t_chans: list of input channels
        t_ok: assumed output channel
        t_list: list of tests to run
    """
    colors = t_keys.get('colors')
    ranges = t_keys.get('ranges')
    # Blend all input tiles
    t_out = mem.tile(t_chans, colors, ranges)
    t_pair = t_ok, t_out

    # Run standard tests by default
    if not t_list:
        t_list = [
            Visual.sanity_check,
            Visual.visual_check,
        ]
    for t_fn in t_list:
        t_fn(t_pair, t_id)


def easy_test_tile(t_r, t_c, t_in, t_id, t_list=None):
    """ Combine one channel to expected output and compare

    Arguments:
        t_r: 2 min,max float32
        t_c: 3 b,g,r float32
        t_id: str name of test
        t_in: input channel
        t_list: list of tests to run
    """
    t_keys = {
        'ranges': t_r[np.newaxis],
        'colors': t_c[np.newaxis],
        'shape': t_in.shape
    }
    t_ok = Visual.u8_colorize(Visual.u8_cut_norm(t_in, t_r), t_c)
    generic_test_tile([t_in], t_id, t_keys, t_ok, t_list)


def many_test_tile(ranges, colors, t_chans, t_id, t_list=None):
    """ Combine many channels to expected output and compare

    Arguments:
        ranges: N channels by 2 min,max float32
        colors: N channels by 3 b,g,r float32
        t_id: str name of test
        t_chans: list of input channels
        t_list: list of tests to run
    """
    t_keys = {
        'ranges': ranges,
        'colors': colors,
        'shape': t_chans[0].shape
    }
    t_ok = Visual.u8_cut_norm_mean(t_chans, ranges, colors)
    generic_test_tile(t_chans, t_id, t_keys, t_ok, t_list)

# _________________________
# Actual entrypoints


def test_tile_1channel_gray():
    """ 1 channel cut and color
    """
    range_all = Visual.fro(0, 100)
    range_hi = Visual.fro(50, 100)
    range_lo = Visual.fro(0, 50)

    # START TEST
    t_id = '1channel_gray_all'
    t_in = Visual.u16_all
    # Check mapping all values to white
    easy_test_tile(range_all, Visual.white, t_in, t_id)
    del (t_in, t_id)

    # START TEST
    t_id = '1channel_gray_0to50'
    t_in = Visual.u16_all
    # Check mapping low values to white
    easy_test_tile(range_lo, Visual.white, t_in, t_id)
    del (t_in, t_id)

    # START TEST
    t_id = '1channel_green_50to100'
    t_in = Visual.u16_all
    # Check mapping high values to green
    easy_test_tile(range_hi, Visual.green, t_in, t_id)
    del (t_in, t_id)


def test_tile_2channel_chess():
    """ 2 channel cut and color
    """
    full_ranges = np.stack((Visual.fro(0, 100),)*2)
    by_colors = np.stack((Visual.blue, Visual.yellow))
    # START TEST
    t_id = '2channel_chess'
    t_chans = [
        Visual.u16_chess0,
        Visual.u16_chess1,
    ]
    # Make sure blue/yellow grid has no overlaps
    many_test_tile(full_ranges, by_colors, t_chans, t_id)
    del (t_chans, t_id)


def main():
    test_tile_1channel_gray()
    test_tile_2channel_chess()
