""" Help load yaml config file
"""
import yaml
import numpy as np


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
