from minsc.blend.mem import tile
from ..constants import Key
from ..constants import Log
import numpy as np


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
    # Test intermediate images
    u16_0to50 = Key.u8_cut_norm(u16_all, Key.fro(0, 50))
    u16_50to100 = Key.u8_cut_norm(u16_all, Key.fro(50, 100))

    def sanity_check(t_pair, t_id='?'):
        """ Compare basic details of two images

        Arguments:
            t_pair: list of two images to compare
            t_id: identity of test
        """
        type_pair = [x.dtype for x in t_pair]
        shape_pair = [x.shape for x in t_pair]
        # Log messages if results are unexpected
        type_msg = "dtypes differ: truth {}, result {}"
        shape_msg = "shapes differ: truth {}, result {}"

        def type_goal():
            """Assume the same data types
            """
            return len(set(type_pair)) == 1
        Log.assume(type_goal, type_msg, type_pair)

        def shape_goal():
            """Assume the same output shapes
            """
            return not np.subtract(*shape_pair).any()
        Log.assume(shape_goal, shape_msg, shape_pair)

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

    def full_check(t_pair, t_id='?'):
        """ Expect two images to be idential

        Arguments:
            t_pair: two arrays assumed identical
            t_id: identity of test
        """
        # Log if some pixels differ
        full_msg = "pixel at {}y, {}x: truth {}, result {}"
        first_diff = next(Log.diff(*t_pair), None)

        def full_goal():
            """ Assume the results have all the same pixels
            """
            return not np.subtract(*t_pair).any()
        Log.assume(full_goal, full_msg, first_diff)


def generic_test_tile(t_id, t_keys, t_chans, t_ok, t_list=[]):
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
            TestKey.visual_check,
            TestKey.full_check,
        ]
    for t_fn in t_list:
        t_fn(t_pair, t_id)


def test_tile_1channel_gray():
    """ 1 channel cut and color
    """
    # START TEST
    t_id = '1channel_gray_all'
    t_chans = TestKey.u16_all[np.newaxis]
    t_ok = TestKey.u8_colorize(TestKey.u16_all)
    t_keys = {
        'ranges': TestKey.fro(0, 100)[np.newaxis],
        'colors': TestKey.white[np.newaxis],
        'shape': t_chans[0].shape
    }
    generic_test_tile(t_id, t_keys, t_chans, t_ok)
    del (t_id, t_keys, t_chans, t_ok)

    # START TEST
    t_id = '1channel_gray_0to50'
    t_chans = TestKey.u16_all[np.newaxis]
    t_ok = TestKey.u8_colorize(TestKey.u16_0to50)
    t_keys = {
        'ranges': TestKey.fro(0, 50)[np.newaxis],
        'colors': TestKey.white[np.newaxis],
        'shape': t_chans[0].shape
    }
    generic_test_tile(t_id, t_keys, t_chans, t_ok)
    del (t_id, t_keys, t_chans, t_ok)

    # START TEST
    t_id = '1channel_green_50to100'
    t_chans = TestKey.u16_all[np.newaxis]
    t_ok = TestKey.u8_colorize(TestKey.u16_50to100, TestKey.green)
    t_keys = {
        'ranges': TestKey.fro(50, 100)[np.newaxis],
        'colors': TestKey.green[np.newaxis],
        'shape': t_chans[0].shape
    }
    generic_test_tile(t_id, t_keys, t_chans, t_ok)
    del (t_id, t_keys, t_chans, t_ok)


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
    colors = by_colors
    ranges = full_ranges
    t_ok = TestKey.u8_cut_norm_mean(t_chans, ranges, colors)
    t_keys = {
        'ranges': ranges,
        'colors': colors,
        'shape': t_chans[0].shape
    }
    # Combine 3 striped images in varied colors
    generic_test_tile(t_id, t_keys, t_chans, t_ok)
    del (t_id, t_keys, t_chans, t_ok, colors, ranges)
