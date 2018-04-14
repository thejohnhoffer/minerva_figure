import numpy as np
import cv2


def to_f32(img):
    """Scale the dynamic range to 0.0 - 1.0

    Arguments:
    img: an integer image
    """
    n_bits = 8 * img.itemsize
    bit_factor = 1.0 / (2.0 ** n_bits)
    return np.float32(img * bit_factor)


def f32_to_bgr(f_img, color=[1, 1, 1]):
    """ Reshape into a color image

    Arguments:
    f_img: float32 image to reshape
    """
    # Give the image a color dimension
    f_vol = f_img[:, :, np.newaxis]
    f_bgr = np.repeat(f_vol, 3, 2) * color
    return (256*f_bgr).astype(np.uint8)


def tile(all_buffer, **kwargs):
    """blend all channels given
    Arguments:
        all_buffer: list of numpy image channels for a tile

    Keywords:
        ranges: float32 2 by N-channel min, max range
        colors: float32 2 by N-channel b, g, r max color

    Returns:
        uint8 y by x by 3 color BGR image
    """
    all_bgr = kwargs.get('colors', np.float32([(1.0, 1.0, 1.0)]))
    all_thresh = kwargs.get('ranges', np.float32([(0.0, 1.0)]))
    tile_shape = kwargs.get('shape', np.uint16([1024, 1024]))

    # final buffer for blending
    color_tile_shape = tuple(tile_shape) + (3,)
    img_buffer = np.zeros(color_tile_shape, dtype=np.float32)

    # Process all channels
    for c_bgr, c_thresh, c_img in zip(all_bgr, all_thresh, all_buffer):
        img_data = to_f32(c_img)
        # Maximum color for this channel
        avg_factor = 1.0 / len(all_buffer)
        color_factor = c_bgr * avg_factor
        # Fraction of full range
        lowest, highest = c_thresh
        clip_size = highest - lowest
        # Clip the image
        clip_data = np.clip(img_data, lowest, highest)
        norm_data = (clip_data - lowest) / clip_size
        # Add the colored data to the image
        y_shape, x_shape = norm_data.shape
        gray_bgr_data = f32_to_bgr(norm_data, color_factor)
        img_buffer[0:y_shape, 0:x_shape] += gray_bgr_data

    return np.uint8(img_buffer)
