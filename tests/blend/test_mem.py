""" Compare blend results with expected output
"""
import numpy as np
from minerva_scripts.blend.mem import tile
from ..constants import Key


class TestKey(Key):
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
    u16_all = Key.square(u16)
    u16_7x7 = Key.square(u16[:49])
    # On/off grid
    u16_chess0 = np.uint16(u16_7x7 % 2 == 0)*u16[-1]
    u16_chess1 = np.uint16(u16_7x7 % 2 == 1)*u16[-1]
    # Slightly overlapping bars
    u16_bar0 = np.tile(Key.square(u16)[:-2:64].T, 3)
    u16_bar1 = np.tile(Key.square(u16)[1:-1:64].T, 3)
    u16_bar2 = np.tile(Key.square(u16)[2::64].T, 3)

    @staticmethod
    def sanity_check(t_pair):
        """ Compare basic details of two images

        Arguments:
            t_pair: list of two images to compare
        """
        type_pair = [x.dtype for x in t_pair]
        shape_pair = [x.shape for x in t_pair]

        # Assume data type and shape
        assert len(set(type_pair)) == 1
        assert not np.subtract(*shape_pair).any()

    @staticmethod
    def full_check(t_pair):
        """ Expect two images to be idential

        Arguments:
            t_pair: two arrays assumed identical
        """
        # Assume pixel-for-pixel image
        assert not np.subtract(*t_pair).any()


def generic_test_tile(t_chans, t_id, t_keys, t_ok, t_list=None):
    """ Run test on tile blend function

    Arguments:
        t_id: str name of test
        t_keys: keywords for call
        t_chans: list of input channels
        t_ok: assumed output channel
        t_list: list of tests to run
    """
    # Blend all input tiles
    t_out = tile(t_chans, **t_keys)
    t_pair = t_ok, t_out

    # Run standard tests by default
    if not t_list:
        t_list = [
            TestKey.sanity_check,
            TestKey.full_check,
        ]
    for t_fn in t_list:
        t_fn(t_pair)


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
    t_ok = TestKey.u8_colorize(TestKey.u8_cut_norm(t_in, t_r), t_c)
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
    t_ok = TestKey.u8_cut_norm_mean(t_chans, ranges, colors)
    generic_test_tile(t_chans, t_id, t_keys, t_ok, t_list)

# _________________________
# Actual pytest entrypoints


def test_tile_1channel_gray():
    """ 1 channel cut and color
    """
    range_all = TestKey.fro(0, 100)
    range_hi = TestKey.fro(50, 100)
    range_lo = TestKey.fro(0, 50)

    # START TEST
    t_id = '1channel_gray_all'
    t_in = TestKey.u16_all
    # Check mapping all values to white
    easy_test_tile(range_all, TestKey.white, t_in, t_id)
    del (t_in, t_id)

    # START TEST
    t_id = '1channel_gray_0to50'
    t_in = TestKey.u16_all
    # Check mapping low values to white
    easy_test_tile(range_lo, TestKey.white, t_in, t_id)
    del (t_in, t_id)

    # START TEST
    t_id = '1channel_green_50to100'
    t_in = TestKey.u16_all
    # Check mapping high values to green
    easy_test_tile(range_hi, TestKey.green, t_in, t_id)
    del (t_in, t_id)


def test_tile_2channel_chess():
    """ 2 channel cut and color
    """
    full_ranges = np.stack((TestKey.fro(0, 100),)*2)
    by_colors = np.stack((TestKey.blue, TestKey.yellow))
    # START TEST
    t_id = '2channel_chess'
    t_chans = [
        TestKey.u16_chess0,
        TestKey.u16_chess1,
    ]
    # Make sure blue/yellow grid has no overlaps
    many_test_tile(full_ranges, by_colors, t_chans, t_id)
    del (t_chans, t_id)
