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

    def sanity_check(tile_pair):
        """ Compare basic details of two images

        Arguments:
            tile_pair: list of two images to compare
        """
        type_pair = [x.dtype for x in tile_pair]
        shape_pair = [x.shape for x in tile_pair]
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

    def visual_check(tile_pair, pre='a'):
        """ Visually compare two images

        Arguments:
            tile_pair: list of two images to compare
            pre: prefix for output images
        """
        # Write out actual images
        diff_image = np.subtract(*tile_pair)
        Log.write_image(tile_pair[0], pre+'_ok')
        Log.write_image(tile_pair[1], pre+'_out')
        Log.write_image(diff_image, pre+'_diff')

    def full_check(tile_pair):
        """ Expect two images to be idential

        Arguments:
            tile_pair: two arrays assumed identical
        """
        # Log if some pixels differ
        full_msg = "pixel at {}y, {}x: truth {}, result {}"
        first_diff = next(Log.diff(*tile_pair), None)

        def full_goal():
            """ Assume the results have all the same pixels
            """
            return not np.subtract(*tile_pair).any()
        Log.assume(full_goal, full_msg, first_diff)


def generic_test_tile(test_id, test_keys, tiles_in, tile_ok):
    """ Run test on tile blend function

    Arguments:
        test_id: str name of test
        test_keys: keywords for call
        tiles_in: list of input channels
        tile_ok: assumed output channel
    """
    # Blend all input tiles
    tile_out = tile(tiles_in, **test_keys)
    tile_pair = tile_ok, tile_out

    # Rough and fine checks
    TestKey.sanity_check(tile_pair)
    TestKey.visual_check(tile_pair, test_id)
    TestKey.full_check(tile_pair)


def test_tile_1channel_gray():
    """ 1 channel cut and color
    """
    # START TEST
    test_id = '1channel_gray_all'
    tiles_in = TestKey.u16_all[np.newaxis]
    tile_ok = TestKey.u8_colorize(TestKey.u16_all)
    test_keys = {
        'ranges': TestKey.fro(0, 100)[np.newaxis],
        'colors': TestKey.white[np.newaxis],
        'shape': tiles_in[0].shape
    }
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok)

    # START TEST
    test_id = '1channel_gray_0to50'
    tiles_in = TestKey.u16_all[np.newaxis]
    tile_ok = TestKey.u8_colorize(TestKey.u16_0to50)
    test_keys = {
        'ranges': TestKey.fro(0, 50)[np.newaxis],
        'colors': TestKey.white[np.newaxis],
        'shape': tiles_in[0].shape
    }
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok)

    # START TEST
    test_id = '1channel_green_50to100'
    tiles_in = TestKey.u16_all[np.newaxis]
    tile_ok = TestKey.u8_colorize(TestKey.u16_50to100, TestKey.green)
    test_keys = {
        'ranges': TestKey.fro(50, 100)[np.newaxis],
        'colors': TestKey.green[np.newaxis],
        'shape': tiles_in[0].shape
    }
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok)


def test_tile_2channel_chess():
    """ 2 channel cut and color
    """
    full_ranges = np.stack((TestKey.fro(0,100),)*2)
    by_colors = np.stack((TestKey.blue, TestKey.yellow))
    # START TEST
    test_id = '2channel_chess'
    tiles_in = [
        TestKey.u16_chess0,
        TestKey.u16_chess1,
    ]
    colors = by_colors
    ranges = full_ranges
    tile_ok = TestKey.u8_cut_norm_mean(tiles_in, ranges, colors)
    test_keys = {
        'ranges': ranges,
        'colors': colors,
        'shape': tiles_in[0].shape
    }
    # Combine 3 striped images in varied colors
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok, colors, ranges)
