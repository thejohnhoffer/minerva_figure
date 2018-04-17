import numpy as np


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


def tile(all_img, colors, ranges=None):
    """blend all channels given
    Arguments:
        all_img: list of numpy image channels for a tile
        colors: N-channel by b, g, r float32 color
        ranges: N-channel by min, max float32 range

    Returns:
        uint8 y by x by 3 color BGR image
    """
    n_chan = len(list(zip(colors, all_img)))
    t_shape = np.amax([a.shape for a in all_img], 0)
    t_shape_color = tuple(t_shape) + (3,)
    # Default range to 0,1 for all channels
    if ranges is None:
        ranges = np.float32(([0, 1],)*n_chan)

    # final buffer for blending
    img_buffer = np.zeros(t_shape_color, dtype=np.float32)

    # Process as many channesl as have colors and ranges
    for c_color, c_range, c_img in zip(colors, ranges, all_img):
        img_data = to_f32(c_img)
        # Maximum color for this channel
        avg_factor = 1.0 / len(all_img)
        color_factor = c_color * avg_factor
        # Fraction of full range
        lowest, highest = c_range
        clip_size = highest - lowest
        # Clip the image
        clip_data = np.clip(img_data, lowest, highest)
        norm_data = (clip_data - lowest) / clip_size
        # Add the colored data to the image
        y_shape, x_shape = norm_data.shape
        gray_bgr_data = f32_to_bgr(norm_data, color_factor)
        img_buffer[0:y_shape, 0:x_shape] += gray_bgr_data

    return np.uint8(img_buffer)
