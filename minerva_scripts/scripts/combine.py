import minerva_scripts.load.disk
import minerva_scripts.blend.mem
import numpy as np
import datetime
import argparse
import yaml
import cv2
import sys
import os

def safe_yaml(y):
    if isinstance(y, dict):
        return {str(k):safe_yaml(y[k]) for k in y}
    if isinstance(y, (list, tuple, set, np.ndarray)):
        return [safe_yaml(v) for v in y]
    if isinstance(y, np.generic):
        return y.item()
    return y

def log_yaml(i, y):
    pretty = {
        'default_flow_style': False,
        'allow_unicode': True,
        'encoding': 'utf-8',
    }
    out = safe_yaml(y)
    if i is not None:
        out = {str(i):out}
    y_str = yaml.dump(out, **pretty)
    print(y_str.decode(encoding='UTF-8'))

def load_config(yml_path, main_key="main"):
    """ Loads a key from a yaml file
    """
    try:
        with open(yml_path, 'r') as yf:
            yml = yaml.load(yf)
            yml_keys = list(yml.keys())
            main_entry = yml[main_key]
            # We read yaml sucessfully
            return main_entry
    except yaml.parser.ParserError as e:
        log_yaml(type(e).__name__, yml_path)
    except (AttributeError, KeyError, TypeError)as e:
        log_yaml('Missing "{}" key'.format(main_key), {
            'keys': yml_keys,
            'yaml': yml_path,
        })
    except IOError as e:
        log_yaml('IOError', e.strerror)

    return None

def parse_config(config_path):
    """
    main:
        IN: {DIR:*, NAME:*}
        OUT: {DIR:*, NAME:*, NOW*}
        RANGES: [[*,*]..]
        COLORS: [[*,*]..]
        TIME: *
        LOD: *

    Arguments:
        config_path: path to yaml with above keys

    Returns:
        t: integer timestep
        l: integer power-of-2 level-of-detail
        r: float32 N channels by 2 min,max
        c: float32 N channels by 3 b,g,r
        o: full output format
        i: full input format
    """
    cfg_data = load_config(config_path)
    if cfg_data is None:
        # Allow empty config file
        cfg_data = {}
    terms = {}

    # Read root values from config
    in_args = cfg_data.get('IN', {})
    out_args = cfg_data.get('OUT', {})
    terms['t'] = int(cfg_data.get('TIME', 0))
    terms['l'] = int(cfg_data.get('LOD', 0))

    # Validate the threshholds and colors
    terms['r'] = np.float32(cfg_data.get('RANGES', [[0,1]]))
    terms['c'] = np.float32(cfg_data.get('COLORS', [[1,1,1]]))

    # Read the paths with defaults
    in_dir = in_args.get('DIR', '~/tmp/minerva_scripts/in')
    out_dir = out_args.get('DIR', '~/tmp/minerva_scripts/out')
    in_name = in_args.get('NAME', '{}_{}_{}_{}_{}_{}.png')
    out_name = out_args.get('NAME', '{}_{}_{}_{}_{}.png')
    # Output stored to current date and time
    now_date = datetime.datetime.now()
    now_time = now_date.time()
    NOW = "{0:04d}_{1:02d}_{2:02d}{4}{3:02d}".format(*[
        now_date.year,
        now_date.month,
        now_date.day,
        now_time.hour,
        os.sep,
    ])
    out_date = out_args.get('NOW', NOW)

    # Format the full paths properly
    out_dir = out_dir.format(NOW=out_date)
    terms['o'] = os.path.join(out_dir, out_name)
    terms['i'] = os.path.join(in_dir, in_name)

    # Create output directory if nonexistant
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return terms

def main(args=sys.argv[1:]):

    # Set up the argument parser
    helps = {
        "main": "combine channels for all tiles",
        "config": """main:
    IN: {DIR:*, NAME:*}
    OUT: {DIR:*, NAME:*, NOW*}
    RANGES: [[*,*]..]
    COLORS: [[*,*]..]
    TIME: *
    LOD: *
    """,
    }
    cmd = argparse.ArgumentParser(description=helps['main'])
    cmd.add_argument('config', nargs='*', default='config.yaml', help=helps['main'])
    parsed = cmd.parse_args(args)

    terms = parse_config(parsed.config)
    log_yaml("parameters", terms)

    # Full path format of input files
    in_path_format = terms['i']
    out_path_format = terms['o']
    # Important parameters
    ALL_RANGES = terms['r']
    ALL_COLORS = terms['c']
    TIME = terms['t']
    LOD = terms['l']

    # Find range of image tiles
    ctlzyx_shape, tile_shape = load.disk.index(in_path_format)
    zyx_shape = ctlzyx_shape[-3::]
    n_channel = ctlzyx_shape[0]

    # Process all z,y,x tiles
    for i in range(np.prod(zyx_shape)):
        z,y,x = np.unravel_index(i, zyx_shape)

        # DERP
        if z != 0:
            continue

        # from disk, load all channels for tile
        all_buffer = load.disk.tile(TIME, LOD, z, y, x, **{
            'format': in_path_format,
            'count': n_channel,
        })

        # from memory, blend all channels loaded
        img_buffer = blend.mem.tile(all_buffer, **{
            'ranges': ALL_RANGES,
            'shape': tile_shape,
            'colors': ALL_COLORS,
        })

        # Write the image buffer to a file
        out_file = out_path_format.format(TIME, LOD, z, y, x)
        try:
            cv2.imwrite(out_file, img_buffer)
        except Exception as e:
            print (e)

if __name__ == "__main__":
    main(sys.argv)
