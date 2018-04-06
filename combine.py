import numpy as np
import cv2
import os

# Load a single image file
def load_file(fmt, args):
    """ Load a single file by pattern

    Args:
        fmt: string defining file pattern
        args: tuple completing pattern

    Returns:
        numpy array loaded from file
    """
    img_name = fmt.format(*args)
    if os.path.exists(img_name):
        return cv2.imread(img_name, 0)
    return None

# Load all the image files in a range
def load_files(fmt, all_vars, n_const=0):
    """ Load a bunch of files

    Args:
        fmt: string defining file pattern
        all_vars: tuple of variables for pattern
        n_const: number of constant variables
    """

    if n_const == len(all_vars):
        yield load_file(fmt, all_vars)
    else:
        n_new = len(all_vars) - n_const
        temp_vars = list(all_vars[:n_const] + (0,)*n_new)
        while True:
            temp_vars[-n_new] += 1
            next_vars = tuple(temp_vars)
            next_file = load_files(fmt, next_vars, n_const-1)
            if next_file is not None:
                yield next_data
                break
            return None


if __name__ == "__main__":

    # Constants
    IN_DIR = "/media/john/420D-AC8E/cycif_images/40BP_59/tiles/"
    IN_NAME_FORMAT = "C{:d}-T{:d}-Z{:d}-L{:d}-Y{:d}-X{:d}.png"
    IN_NAME_NUM_VARS = 6
    
    # Full path format of input files
    full_format = os.path.join(IN_DIR, IN_NAME_FORMAT)
    all_variables = (0,) * IN_NAME_NUM_VARS

    # Load all input files
    for fi in iter(load_files(full_format, all_variables)):
        print(img)
