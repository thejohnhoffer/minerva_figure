from minsc.blend.mem import tile
import numpy as np
import cv2


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
        sq_shape = np.ceil(np.sqrt((vec.shape[0], -1)))
        return np.reshape(vec, sq_shape.astype(vec.dtype))

    def resize(img, scalar):
        """ Scale an image by a scalar

        Arguments
        img: image to scale
        scalar: scale factor
        """
        kwargs = {
            'interpolation': cv2.INTER_NEAREST,
            'fx': scalar,
            'fy': scalar,
        }
        return cv2.resize(img, **kwargs)

    def sq_scale(vec, scalar):
        """ Reshape into square and resize

        Arguments
            vec: flattened image
            scalar: scale factor
        """
        return Key.resize(Key.square(vec), scalar)


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
    img256_grays = Key.to_bgr(Key.sq_scale(Key.all_u8, 2))


def test_tile__1channel_gray():
    """ 1 channel map to white
    """
    keywords = {
        'ranges': TestKey.range_full[np.newaxis],
        'colors': TestKey.color_white[np.newaxis],
    }
    # All u16 grays to all u8 gray bgr
    gray_in = TestKey.img256_u16[np.newaxis]
    gray_bgr_out = TestKey.img256_grays

    # u8 output from 1-channel u16 input
    assert gray_bgr_out == tile(gray_in, **keywords)
