from minsc.blend.mem import tile
import numpy as np
import inspect
import cv2
import sys
import os


class Key(object):
    """ Used to derive test constants
    """
    all_u16 = np.uint16(range(2**16))
    all_u8 = np.uint8(range(2**8))

    def to_bgr(img, color=[1, 1, 1]):
        """ Reshape into a color image

        Arguments
        img: the image to reshape
        """
        sq = img[:, :, np.newaxis]
        bgr_sq = np.repeat(sq, 3, 2) * color
        return bgr_sq.astype(np.uint8)

    def square(vec):
        """ Reshape a vector into an square image
        """
        sq_shape = np.ceil(np.sqrt((vec.shape[0],)*2))
        return np.reshape(vec, sq_shape.astype(vec.dtype))


class TestKey(Key):
    """ Constants used for testing
    """
    # Test sample ranges and colors
    range_full = np.float32([0, 1])
    color_white = np.float32([1, 1, 1])
    # Test input images
    img16_u8 = Key.square(Key.all_u8)
    img256_u16 = Key.square(Key.all_u16)
    # Test output images
    img256_grays = Key.to_bgr(Key.square(Key.all_u16)/256)

    def diff(a1,a2):
        """ Iterate nonzero indices of images

        Returns
            y index of pixel
            x index of pixel
            value of pixel in a1
            value of pixel in a2
        """
        height = min(a1.shape[0], a2.shape[0])
        width = min(a1.shape[1], a2.shape[1])
        # Iterate through the image pixels
        for yi in range(height):
            for xi in range(width):
                diff = a2[yi,xi] - a1[yi,xi]
                if not any(diff):
                    continue
                # Yield different pixels
                yield yi,xi,a1[yi,xi],a2[yi,xi]

    def assume(condition, msg, args):
        """ Log msg with args if not condition

        Arguments:
            condtion: function should resolve true
            msg: string to format with args
            args: values to format string
        """
        statement = inspect.getsource(condition)
        try:
            result = condition()
            assert result
        except AssertionError as a_e:
            print(statement.strip(), file=sys.stderr)
            if args:
                print(msg.format(*args), file=sys.stderr)
            raise a_e

    def write_image(a,name='tmp'):
        """ write array to temp image file
        """
        f_name = name + '.png'
        f_root = os.path.dirname(__file__)
        f_path = inspect.stack()[1].function
        tmp_path = os.path.join(f_root, 'tmp', f_path)
        tmp_png = os.path.join(tmp_path, f_name)
        # write image with temp name
        print('writing {}'.format(f_name))
        try:
            os.makedirs(tmp_path)
            os.remove(tmp_png)
        except OSError:
            pass
        cv2.imwrite(tmp_png, a)

def test_tile__1channel_gray():
    """ 1 channel map to white
    """
    # All u16 grays to all u8 gray bgr
    gray_in = TestKey.img256_u16[np.newaxis]
    gray_bgr_ok = TestKey.img256_grays
    # Additional arguments to blend tile
    keywords = {
        'ranges': TestKey.range_full[np.newaxis],
        'colors': TestKey.color_white[np.newaxis],
        'shape': gray_in[0].shape,
    }

    # u8 output from 1-channel u16 input
    gray_bgr_out = tile(gray_in, **keywords)
    tile_pair = gray_bgr_ok, gray_bgr_out
    type_pair = [x.dtype for x in tile_pair]
    shape_pair = [x.shape for x in tile_pair]
    # Log messages if results are unexpected
    type_msg = "dtypes differ: truth {}, result {}"
    shape_msg = "shapes differ: truth {}, result {}"
    full_msg = "pixel at {}y,{}x: truth {}, result {}"

    # Assume the same data types
    type_goal = lambda: len(set(type_pair)) == 1
    TestKey.assume(type_goal, type_msg, type_pair)
    # Assume the same output shapes
    shape_goal = lambda: not np.subtract(*shape_pair).any()
    TestKey.assume(shape_goal, shape_msg, shape_pair)
    # Write out the actual diff
    diff_image = np.subtract(*tile_pair)
    TestKey.write_image(gray_in[0], 'a_in')
    TestKey.write_image(gray_bgr_ok, 'a_ok')
    TestKey.write_image(gray_bgr_out, 'a_out')
    TestKey.write_image(diff_image, 'a_diff')
    # Assume the same output tiles
    first_diff = next(TestKey.diff(*tile_pair), None)
    full_goal = lambda: not np.subtract(*tile_pair).any()
    TestKey.assume(full_goal, full_msg, first_diff)
