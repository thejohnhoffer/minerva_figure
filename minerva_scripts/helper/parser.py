""" Help process inputs
"""
import os


def real_path(path, root=''):
    """ Read a path relative to root directory

    Arguments:
        path: The path to parse
        root: The paretn of the path
    """
    user_path = os.path.expanduser(path)
    root_path = os.path.join(root, user_path)
    return os.path.abspath(root_path)
