''' Test to crop all tiles in a region
'''
import argparse
import pathlib
import urllib
import struct
import json
import sys
import ssl
import re
import os
import io

import skimage.io
import skimage.exposure
import numpy as np
from minerva_lib import crop

######
# Omero API
###

HEADERS = {
    'Cookie': os.environ['OME_COOKIE'],
}

ssl._create_default_https_context = ssl._create_unverified_context


class omero():

    @staticmethod
    def image(c, limit, *args):
        ''' Load a single channel by pattern

        Args:
            c: zero-based channel index
            limit: max image pixel value
            args: tuple completing pattern

        Returns:
            numpy array loaded from file
        '''

        def format_channel(c):
            api_c = c + 1
            selected = '{}|0:{}$000000'.format(api_c, limit)
            filler = [str(-i) for i in range(1, api_c)] + ['']
            return ','.join(filler) + selected

        url = 'https://omero.hms.harvard.edu/webgateway/render_image_region/'
        url += '{}/{}/{}/?m=g&format=tif&tile={},{},{}'.format(*args)
        url += '&c=' + format_channel(c)
        print(url)

        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req) as response:
                f = io.BytesIO(response.read())
                return skimage.io.imread(f)[:, :, 0]
        except urllib.error.HTTPError as e:
            print(e)
            return None

        return None

    @staticmethod
    def index(image_id):
        '''Find all the file paths in a range

        Args:
            image_id: the id of image in omero

        Returns:
            indices: size in channels, times, LOD, Z, Y, X
            tile: image tile size in pixels: y, x
            limit: max image pixel value
        '''
        config = {}
        url = 'https://omero.hms.harvard.edu/webgateway/'
        url += 'imgData/{}'.format(image_id)
        print(url)

        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req) as response:
            config = json.loads(response.read())

        dtype = config['meta']['pixelsType']
        tw, th = map(config['tile_size'].get,
                     ('width', 'height'))
        w, h, c, t, z = map(config['size'].get,
                            ('width', 'height', 'c', 't', 'z'))
        y = int(np.ceil(h / th))
        x = int(np.ceil(w / tw))

        return {
            'limit': np.iinfo(getattr(np, dtype)).max,
            'levels': config['levels'],
            'tile_size': [th, tw],
            'ctxy': [c, t, x, y],
        }


class api():

    @staticmethod
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
            c: float32 N channels by 3 red, green, blue
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
            'ctxy': meta['ctxy'],
            'limit': meta['limit'],
            'levels': meta['levels'],
            'tile_size': meta['tile_size'],
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


######
# Minerva API
###

def format_input(args):
    ''' Combine all parameters
    '''
    id_, color_, range_ = args
    return {
        'channel': id_,
        'color': color_,
        'min': range_[0],
        'max': range_[1],
    }


def do_crop(load_tile, channels, tile_size, full_origin, full_size,
            levels=1, max_size=2000):
    ''' Interface with minerva_lib.crop

    Args:
        load_tile: Function to supply 2D numpy array
        channels: List of dicts of channel rendering settings
        tile_size: The width, height of a single tile
        full_origin: Request's full-resolution x, y origin
        full_size: Request's full-resolution width, height
        levels: The number of pyramid levels
        max_size: The maximum response width or height

    Returns:
        2D numpy float array of with width, height given by
        `full_size` if `full_size <= max_size` or width, height given by
        `full_size / 2 ** l <= max_size` for the lowest `l` meeting
        `l < levels`. The array is a composite of all channels
        for full or partial tiles within `full_size` from `full_origin`.
    '''

    level = crop.get_optimum_pyramid_level(full_size, levels, max_size)
    crop_origin = crop.scale_by_pyramid_level(full_origin, level)
    crop_size = crop.scale_by_pyramid_level(full_size, level)
    print(f'Cropping 1/{level} scale')

    image_tiles = []

    for channel in channels:

        (red, green, blue) = channel['color']
        _id = channel['channel']
        _min = channel['min']
        _max = channel['max']

        for indices in crop.select_tiles(tile_size, crop_origin, crop_size):

            (i, j) = indices

            # Disallow negative tiles
            if i < 0 or j < 0:
                continue

            # Load image from Omero
            image = load_tile(_id, level, i, j)

            # Disallow empty images
            if image is None:
                continue

            # Add to list of tiles
            image_tiles.append({
                'min': _min,
                'max': _max,
                'image': image,
                'indices': (i, j),
                'color': (red, green, blue),
            })

    return crop.stitch_tiles(image_tiles, tile_size, crop_origin, crop_size)


######
# Entrypoint
###

def main(args):
    """ Crop a region
    """
    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description="Crop a region"
    )

    default_url = '/548111/0/0/?c=1|0:65535$FF0000,3|0:65535$0000FF'
    default_url += '&region=-8160,-16416,48960,48960'
    cmd.add_argument(
        'url', nargs='?', default=default_url,
        help='OMERO.figure render_scaled_region url'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help='output directory'
    )

    parsed = cmd.parse_args(args)
    out_file = str(pathlib.Path(parsed.o, 'out.png'))

    # Read parameters from URL and API
    keys = api.scaled_region(parsed.url)

    # Make array of channel parameters
    inputs = zip(keys['chan'], keys['c'], keys['r'])
    channels = map(format_input, inputs)

    # OMERO loads the tiles
    def ask_omero(c, l, i, j):
        return omero.image(c, keys['limit'], keys['iid'],
                           keys['z'], keys['t'], l, i, j)

    # Minerva does the cropping
    out = do_crop(ask_omero, channels, keys['tile_size'],
                  keys['origin'], keys['shape'], keys['levels'],
                  keys['max_size'])

    # Write the image buffer to a file
    try:
        skimage.io.imsave(out_file, np.uint8(255 * out))
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main(sys.argv[1:])
