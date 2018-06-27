""" Help load yaml config file
"""
import yaml
import pathlib
import numpy as np
import os


def safe_yaml(y_val):
    """ Handle numpy values when printing yaml

    Arguments:
        y_val: yaml object
    """
    if isinstance(y_val, dict):
        return {str(k): safe_yaml(y_val[k]) for k in y_val}
    if isinstance(y_val, (list, tuple, set, np.ndarray)):
        return [safe_yaml(v) for v in y_val]
    if isinstance(y_val, np.generic):
        return y_val.item()
    return y_val


def log_yaml(i, y_val):
    """ Print a yaml file to standard out

    Arguments:
        i: Label of whole file
        y_val: content of whole file
    """
    pretty = {
        'default_flow_style': False,
        'allow_unicode': True,
        'encoding': 'utf-8',
    }
    out = safe_yaml(y_val)
    if i is not None:
        out = {str(i): out}
    y_str = yaml.dump(out, **pretty)
    print(y_str.decode(encoding='UTF-8'))


def load_yaml(yml_path, main_key="main"):
    """ Loads a key from a yaml file

    Arguments:
        yml_path: The path to the file
        main_key: The key to load from the file
    """
    yml_keys = []
    try:
        with open(yml_path, 'r') as y_f:
            yml = yaml.load(y_f)
            yml_keys = list(yml.keys())
            main_entry = yml[main_key]
            # We read yaml sucessfully
            return main_entry
    except yaml.parser.ParserError as p_e:
        log_yaml(type(p_e).__name__, yml_path)
        print(p_e)
    except (AttributeError, KeyError, TypeError):
        log_yaml('Missing "{}" key'.format(main_key), {
            'keys': yml_keys,
            'yaml': yml_path,
        })
    except IOError as i_e:
        log_yaml('IOError', i_e.strerror)

    return {}


def parse_main(config):
    """
    main:
        CHANNELS: [0, 1..]
        RANGES: [[*, *]..]
        COLORS: [[*, *]..]
        TIME: *
        LOD: *

    Arguments:
        config: path to yaml with above keys

    Return Keywords:
        t: integer timestep
        chan: integer N channels by 1 index
        l: integer power-of-2 level-of-detail
        r: float32 N channels by 2 min, max
        c: float32 N channels by 3 r, g, b
    """

    terms = {}
    cfg_data = {}

    if config:
        data = load_yaml(config)
        cfg_data = data if data else {}

    # Read root values from config
    terms['t'] = int(cfg_data.get('TIME', 0))
    terms['l'] = int(cfg_data.get('LOD', 0))

    # Validate the threshholds and colors
    terms['r'] = np.float32(cfg_data.get('RANGES', [[0, 1]]))
    terms['c'] = np.float32(cfg_data.get('COLORS', [[1, 1, 1]]))
    n_channel = min(map(len, map(terms.get, 'rc')))
    terms['r'] = terms['r'][:n_channel]
    terms['c'] = terms['c'][:n_channel]

    # Set order of channels
    default_order = np.arange(n_channel, dtype=np.uint16)
    terms['chan'] = cfg_data.get('CHANNELS', default_order)

    return terms


def parse(key='main', **kwargs):
    """
    Arguments:
        key: key for yaml config file

    Keyword Arguments:
        config: path to yaml config file
        o: output directory
        i: input directory

    Returns:
        configured terms with defaults
    """
    in_name = 'C{0:}-T{1:}-Z{3:}-L{2:}-Y{4:}-X{5:}.png'
    out_name = 'T{0:}-Z{2:}-L{1:}-Y{3:}-X{4:}.png'

    terms = parse_main(kwargs.get('config', ''))

    # Read the paths with defaults
    in_dir = kwargs.get('i', '')
    out_dir = kwargs.get('o', '')

    # Join the full paths properly
    terms['i'] = str(pathlib.Path(in_dir, in_name))
    terms['o'] = str(pathlib.Path(out_dir, out_name))

    # Create output directory if nonexistant
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return terms
