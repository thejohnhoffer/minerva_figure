from minsc.blend.mem import tile
from ..constants import Key
from ..constants import Log
import numpy as np


class TestKey(Key):
    """ Constants used for testing
    """
    # Test sample ranges and colors
    range_full = np.float32([0, 1])
    range_0to50 = np.float32([0, 0.5])
    color_white = np.float32([1, 1, 1])
    # Test input images
    u16_mono = Key.square(Key.all_u16)
    # Test output images
    u16_grays = Key.to_bgr(Key.square(Key.all_u16))

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
    """ 1 channel map to white
    """
    # START TEST
    # Input u16 grays, expect all u8 gray bgr
    tiles_in = TestKey.u16_mono[np.newaxis]
    tile_ok = TestKey.u16_grays
    test_id = '1channel_gray'
    test_keys = {
        'ranges': TestKey.range_full[np.newaxis],
        'colors': TestKey.color_white[np.newaxis],
        'shape': tiles_in[0].shape
    }
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok)

    # START TEST
    test_id = '1channel_gray_0to50'
    test_range = TestKey.range_0to50
    tiles_in = TestKey.u16_mono[np.newaxis]
    tile_ok = TestKey.norm_cut(TestKey.u16_grays, test_range)
    test_keys = {
        'ranges': test_range[np.newaxis],
        'colors': TestKey.color_white[np.newaxis],
        'shape': tiles_in[0].shape
    }
    generic_test_tile(test_id, test_keys, tiles_in, tile_ok)
    del (test_id, test_keys, tiles_in, tile_ok)
