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

    def norm_cut(img, r):
        """ Cut and normalize image to range

        Arguments:
            img: the integer image to cut
            r: the min and max used to cut

        Returns:
            uint8 cut and normalized image
        """
        f_img = Key.to_float(img)
        cut_img = f_img * (f_img >= r[0]) * (f_img < r[1])
        norm_img = (cut_img - r[0]) / np.diff(r)
        return np.uint8(256*norm_img)

    def to_float(img):
        """Scale the dynamic range to 0.0 - 1.0

        Arguments:
        img: an integer image
        """
        n_values = np.iinfo(img.dtype).max + 1
        return np.float32(img / n_values)

    def to_bgr(img, color=[1, 1, 1]):
        """ Reshape into a color image

        Arguments:
        img: the image to reshape
        """
        f_img = Key.to_float(img)
        # Give the image a color dimension
        f_vol = f_img[:, :, np.newaxis]
        f_bgr = np.repeat(f_vol, 3, 2) * color
        return np.uint8(256*f_bgr)

    def square(vec):
        """ Reshape a vector into an square image
        """
        sq_shape = np.ceil(np.sqrt((vec.shape[0],)*2))
        return np.reshape(vec, sq_shape.astype(vec.dtype))

class Log(object):

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
