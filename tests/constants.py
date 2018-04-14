import numpy as np
import inspect
import cv2
import sys
import os


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


class Log(object):

    def diff(a1, a2):
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
                diff = a2[yi, xi] - a1[yi, xi]
                if not any(diff):
                    continue
                # Yield different pixels
                yield yi, xi, a1[yi, xi], a2[yi, xi]

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

    def write_image(a, name='tmp'):
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
