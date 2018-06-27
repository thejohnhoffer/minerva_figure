""" Help load yaml config file
"""
from parse import parse
import struct


def scaled_region(url):
    """ Just parse the rendered_scaled_region API
    Arguments:
        url: "<matching OMERO.figure API>"

    Return Keywords:
        t: integer timestep
        z: depth in image stack
        region: x, y, width, height
        max_size: the max image width or height
        channels:
            cid: channel id
            shown: whether to use channel
            min: min value (in image range)
            max: max value (in image range)
            color: R,G,B from 0-255
    """
    def parse_channel(c):
        item = '{:d}|{:d}:{:d}${}'
        cid, _min, _max, _hex = parse(item, c).fixed
        hex_bytes = bytearray.fromhex(_hex)
        return {
            'min': _min,
            'max': _max,
            'shown': cid > 0,
            'cid': abs(cid) - 1,
            'color': struct.unpack('BBB', hex_bytes)
        }

    def parse_region(r):
        return parse('{:d},{:d},{:d},{:d}', r).fixed

    fixed = 'render_scaled_region/{:d}/{:d}/{:d}/?{}'
    api = parse('{root}' + fixed, url)
    iid, z, t, query = api.fixed

    # Make parameters dicitonary
    parameters = {}
    for param in query.split('&'):
        key, value = parse('{}={}', param).fixed
        parameters[key] = value

    max_size = parameters.get('max_size', 2000)
    channels = parameters['c'].split(',')
    region = parameters['region']

    # Return parsed URL
    return {
        # Ignore URL components only used in javascript
        # 'maps': json.loads(parameters['maps']),
        # 'm': parameters['m'],

        't': t,
        'z': z,
        'iid': iid,
        'max_size': max_size,
        'region': parse_region(region),
        'channels': list(map(parse_channel, channels))
    }
