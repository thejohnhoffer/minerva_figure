""" Help load yaml config file
"""
import yaml
import pathlib
import numpy as np
import os

from . import api


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

    return None


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


def parse_scaled_region(config):
    """
    render_scaled_region:
        URL: "<matching OMERO.figure API>"
        LIMIT: <maximum integer for range>

    Arguments:
        config: path to yaml with above keys

    Return Keywords:
        t: integer timestep
        origin:
            integer [x, y, z]
        shape:
            [width, height]
        chan: integer N channels by 1 index
        l: integer power-of-2 level-of-detail
        r: float32 N channels by 2 min, max
        c: float32 N channels by 3 r, g, b
    """

    cfg_url = '/render_scaled_region/1337/0/0/?'
    cfg_url += 'c=1|0:65535$0000FF&&region=0,0,512,512'
    cfg_limit = 255

    # Allow config file
    if config:
        key = 'render_scaled_region'
        data = load_yaml(config, key)
        cfg_url = data.get('URL', cfg_url)
        cfg_limit = data.get('LIMIT', cfg_limit)

    def get_range(chan):
        r = np.array([chan['min'], chan['max']])
        return np.clip(r / cfg_limit, 0, 1)

    def get_color(chan):
        c = np.array(chan['color']) / 255
        return np.clip(c, 0, 1)

    # Parse the url
    cfg_data = api.scaled_region(cfg_url)
    x, y, width, height = cfg_data['region']
    longest_side = max(width, height)
    max_size = cfg_data['max_size']

    # Calculate the level of detail
    lod = np.ceil(np.log2(longest_side / max_size))
    shape = np.array([width, height]) / (2 ** lod)
    origin = np.array([x, y, cfg_data['z']]) / (2 ** lod)

    # Get active channels
    channels = cfg_data['channels']
    chan = [c for c in channels if c['shown']]

    return {
        'r': np.array([get_range(c) for c in chan]),
        'c': np.array([get_color(c) for c in chan]),
        'chan': np.int64([c['cid'] for c in chan]),
        'origin': np.int64(np.floor(origin)),
        'shape': np.int64(np.floor(shape)),
        't': cfg_data['t'],
        'l': int(lod)
    }


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

    terms = {
        'main': parse_main,
        'region': parse_scaled_region
    }[key](kwargs.get('config', ''))

    # Read the paths with defaults
    try:
        in_dir = kwargs['i']
        out_dir = kwargs['o']
    except KeyError as k_e:
        raise k_e

    # Join the full paths properly
    terms['i'] = str(pathlib.Path(in_dir, in_name))
    terms['o'] = str(pathlib.Path(out_dir, out_name))

    # Create output directory if nonexistant
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return terms
