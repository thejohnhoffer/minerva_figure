import numpy as np


class Key(object):
    """ Helper functions for unit tests
    """
    def fro(*args):
        """Make vector from percentages
        """
        if len(args) == 1:
            args = args[0]
        return np.float32(args)/100

    def to_f32(img):
        """Scale the dynamic range to 0.0 - 1.0

        Arguments:
            img: an integer image
        """
        try:
            n_values = np.iinfo(img.dtype).max + 1
            return np.float32(img / n_values)
        except ValueError:
            return img

    def to_u8(f_img):
        """Compress the range to 0 - 256

        Arguments:
            f_img: a uint32 image
        """
        return np.uint8(256*f_img)

    def u8_cut_norm_mean(img_list, r_list, c_list):
        """ Cut, normalize, color, and combine images

        Arguments:
            img_list: list of images to process
            r_list: all min and max used to cut
            r_list: all colors used to color

        Returns:
            uint8 cut, normalized average image
        """
        # Clip channels in range
        f32_list = list(map(Key.to_f32, img_list))
        step_0 = zip(f32_list, r_list)
        norm_list = [Key.cut_norm(*s) for s in step_0]
        # Map channels to colors
        step_1 = zip(norm_list, c_list)
        color_list = [Key.colorize(*s) for s in step_1]
        # All channels must have the same shape
        sh = set([x.shape for x in color_list])
        assert len(sh) == 1
        # Return the mean of the colors
        mean_image = np.mean(color_list, 0)
        return Key.to_u8(mean_image)

    def cut_norm(img, r):
        """ Cut and normalize image to range

        Arguments:
            img: the integer image to cut
            r: the min and max used to cut

        Returns:
            float32 cut and normalized image
        """
        f_img = Key.to_f32(img)
        cut_img = f_img * (f_img >= r[0]) * (f_img < r[1])
        return (cut_img - r[0]) / np.diff(r)

    def u8_cut_norm(img, r):
        """ Like cut_norm but returns uint8
        """
        return Key.to_u8(Key.cut_norm(img, r))

    def colorize(img, color=[1, 1, 1]):
        """ Reshape into a color image

        Arguments:
            img: integer image to colorize

        Returns:
            float32 3-channel color image
        """
        f_img = Key.to_f32(img)
        # Give the image a color dimension
        f_vol = f_img[:, :, np.newaxis]
        return np.repeat(f_vol, 3, 2) * color

    def u8_colorize(img, color=[1,1,1]):
        """ Like colorize but returns uint8
        """
        return Key.to_u8(Key.colorize(img, color))

    def square(vec):
        """ Reshape a vector into an square image

        Arguments:
            vec: integer vector
        """
        sq_shape = np.ceil(np.sqrt((vec.shape[0],)*2))
        return np.reshape(vec, sq_shape.astype(vec.dtype))
