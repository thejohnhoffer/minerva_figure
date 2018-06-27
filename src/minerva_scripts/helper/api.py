""" Help load yaml config file
"""
import numpy as np
import struct
import re

from ..load import omero


def scaled_region(url):
    """ Just parse the rendered_scaled_region API
    Arguments:
        url: "<matching OMERO.figure API>"

    Return Keywords:
        iid: image id
        t: integer timestep
        z: integer z position in stack
        max_size: maximum extent in x or y
        origin:
            integer [x, y]
        shape:
            [width, height]
        chan: integer N channels by 1 index
        r: float32 N channels by 2 min, max
        c: float32 N channels by 3 r, g, b
        indices: size in channels, times, LOD, Z, Y, X
        tile: image tile size in pixels: y, x
        limit: max image pixel value
    """

    url_match = re.search('render_scaled_region', url)
    url = url[(lambda x: x.end() if x else 0)(url_match):]
    if url[0] == '/':
        url = url[1:]

    def parse_channel(c):
        cid, _min, _max, _hex = re.split('[:|$]', c)
        hex_bytes = bytearray.fromhex(_hex)
        return {
            'min': int(_min),
            'max': int(_max),
            'shown': int(cid) > 0,
            'cid': abs(int(cid)) - 1,
            'color': struct.unpack('BBB', hex_bytes)
        }

    def parse_region(r):
        return list(map(float, r.split(',')))

    print(url)
    iid, z, t = url.split('?')[0].split('/')[:3]
    query = url.split('?')[1]
    parameters = {}

    # Make parameters dicitonary
    for param in query.split('&'):
        key, value = param.split('=')
        parameters[key] = value

    max_size = parameters.get('max_size', 2000)
    channels = parameters['c'].split(',')
    region = parameters['region']

    # Extract data from API
    channels = list(map(parse_channel, channels))
    x, y, width, height = parse_region(region)

    # Reformat data from API
    chans = [c for c in channels if c['shown']]
    shape = np.array([width, height])
    origin = np.array([x, y])

    # Make API request to interpret url
    meta = omero.index(iid)

    def get_range(chan):
        r = np.array([chan['min'], chan['max']])
        return np.clip(r / meta['limit'], 0, 1)

    def get_color(chan):
        c = np.array(chan['color']) / 255
        return np.clip(c, 0, 1)

    return {
        'tile': meta['tile'],
        'limit': meta['limit'],
        'indices': meta['indices'],
        'r': np.array([get_range(c) for c in chans]),
        'c': np.array([get_color(c) for c in chans]),
        'chan': np.int64([c['cid'] for c in chans]),
        'max_size': int(max_size),
        'origin': origin,
        'shape': shape,
        'iid': int(iid),
        't': int(t),
        'z': int(z)
    }
