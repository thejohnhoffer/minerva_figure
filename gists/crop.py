''' Test to crop all tiles in a region
'''
from functools import reduce
import numpy as np
import skimage.exposure
import skimage.io
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

######
# blend.py
#
# from Minerva Library version 0.0.1
# https://github.com/sorgerlab/minerva-lib-python/
###


def composite_channel(target, image, color, range_min, range_max, out=None):
    ''' Render _image_ in pseudocolor and composite into _target_

    By default, a new output array will be allocated to hold
    the result of the composition operation. To update _target_
    in place instead, specify the same array for _target_ and _out_.

    Args:
        target: Numpy array containing composition target image
        image: Numpy array of image to render and composite
        color: Color as r, g, b float array within 0, 1
        range_min: Threshhold range minimum, float within 0, 1
        range_max: Threshhold range maximum, float within 0, 1
        out: Optional output numpy array in which to place the result.

    Returns:
        A numpy array with the same shape as the composited image.
        If an output array is specified, a reference to _out_ is returned.
    '''

    if out is None:
        out = target.copy()

    # Rescale the new channel to a float64 between 0 and 1
    f64_range = (range_min, range_max)
    f64_image = skimage.img_as_float(image)
    f64_image = skimage.exposure.rescale_intensity(f64_image, f64_range)

    # Colorize and add the new channel to composite image
    for i, component in enumerate(color):
        out[:, :, i] += f64_image * component

    return out


######
# crop.py
#
# Minerva Library PR #21
# https://github.com/sorgerlab/minerva-lib-python/pull/21/files
#
# Pip requirement
# git+https://github.com/thejohnhoffer/minerva-lib-python@crop#minerva-lib
###

class crop():

    @staticmethod
    def get_lod(lods, max_size, width, height):
        ''' Calculate the level of detail below a maximum
        Arguments:
            lods: Number of available levels of detail
            max_size: Maximum output image extent in x or y
            width: Full-resolution extent of image in x
            height: Full-resolution Extent of image in y
        Returns:
            Integer power of 2 level of detail
        '''

        longest_side = max(width, height)
        lod = np.ceil(np.log2(longest_side / max_size))
        return int(np.clip(lod, 0, lods - 1))

    @staticmethod
    def get_lod_1tile(lods, tile_size, width, height):
        ''' Calculate the level of detail to request one tile

        Arguments:
            lods: Number of available levels of detail
            tile_size: width, height of one tile
            width: Full-resolution extent of image in x
            height: Full-resolution Extent of image in y

        Returns:
            Integer power of 2 level of detail
        '''
        return crop.get_lod(lods, min(*tile_size), width, height)

    @staticmethod
    def get_lod_4tiles(lods, tile_size, width, height):
        ''' Calculate the level of detail for at most four tiles

        Arguments:
            lods: Number of available levels of detail
            tile_size: width, height of one tile
            width: Full-resolution extent of image in x
            height: Full-resolution Extent of image in y

        Returns:
            Integer power of 2 level of detail
        '''
        return crop.get_lod(lods, 2 * min(*tile_size), width, height)

    @staticmethod
    def apply_lod(coordinates, lod):
        ''' Apply the level of detail to coordinates

        Arguments:
            coordinates: Coordinates to downscale by _lod_
            lod: Integer power of 2 level of detail

        Returns:
            downscaled integer coordinates
        '''

        scaled_coords = np.array(coordinates) / (2 ** lod)
        return np.int64(np.floor(scaled_coords))

    @staticmethod
    def select_tiles(tile_size, origin, crop_size):
        ''' Select tile coordinates covering crop region

        Args:
            tile_size: width, height of one tile
            origin: x, y coordinates to begin subregion
            crop_size: width, height to select

        Returns:
            List of integer i, j tile indices
        '''
        start = np.array(origin)
        end = start + crop_size
        fractional_start = start / tile_size
        fractional_end = end / tile_size

        # Round to get indices containing subregion
        first_index = np.int64(np.floor(fractional_start))
        last_index = np.int64(np.ceil(fractional_end))

        # Calculate all indices between first and last
        index_shape = last_index - first_index
        offsets = np.argwhere(np.ones(index_shape))
        indices = first_index + offsets

        return indices.tolist()

    @staticmethod
    def get_subregion(indices, tile_size, origin, crop_size):
        ''' Define subregion to select from within tile

        Args:
            indices: integer i, j tile indices
            tile_size: width, height of one tile
            origin: x, y coordinates to begin subregion
            crop_size: width, height to select

        Returns:
            start uv, end uv relative to tile
        '''

        crop_end = np.int64(origin) + crop_size
        tile_start = np.int64(indices) * tile_size
        tile_end = tile_start + tile_size

        return [
            np.maximum(origin, tile_start) - tile_start,
            np.minimum(tile_end, crop_end) - tile_start
        ]

    @staticmethod
    def get_position(indices, tile_size, origin):
        ''' Define position of cropped tile relative to origin

        Args:
            indices: integer i, j tile indices
            tile_size: width, height of one tile
            origin: x, y coordinates to begin subregion

        Returns:
            The xy position relative to origin
        '''

        tile_start = np.int64(indices) * tile_size

        return np.maximum(origin, tile_start) - origin

    @staticmethod
    def stitch_tile(out, subregion, position, tile):
        ''' Position image tile into output array

        Args:
            out: 2D RGB numpy array to contain stitched channels
            subregion: Start uv, end uv to get from tile
            position: Origin of tile when composited in _out_
            tile: 2D numpy array to stitch within _out_

        Returns:
            A reference to _out_
        '''

        # Take subregion from tile
        [u0, v0], [u1, v1] = subregion
        subtile = tile[v0:v1, u0:u1]
        shape = np.int64(subtile.shape)

        # Define boundary
        x0, y0 = position
        y1, x1 = [y0, x0] + shape[:2]

        # Assign subregion within boundary
        out[y0:y1, x0:x1] += subtile

        return out

    @staticmethod
    def stitch_tiles(tiles, tile_size, crop_size, order='before'):
        ''' Position all image tiles for all channels

        Args:
            tiles: Iterator of tiles to blend. Each dict in the
                list must have the following rendering settings:
                {
                    channel: Integer channel index
                    indices: Integer i, j tile indices
                    image: Numpy 2D image data of any type
                    color: Color as r, g, b float array within 0, 1
                    min: Threshhold range minimum, float within 0, 1
                    max: Threshhold range maximum, float within 0, 1
                    subregion: The start uv, end uv relative to tile
                    position: The xy position relative to origin
                }
            tile_size: width, height of one tile
            crop_size: The width, height of output image
            order: Composite `'before'` or `'after'` stitching

        Returns:
            For a given `shape` of `(width, height)`,
            returns a float32 RGB color image with shape
            `(height, width, 3)` and values in the range 0 to 1
        '''
        def stitch(a, t):
            return crop.stitch_tile(a, t['subregion'],
                                    t['position'], t['image'])

        def composite(a, t):
            h, w = t['image'].shape
            return composite_channel(a[:h, :w], t['image'], t['color'],
                                     t['min'], t['max'], a[:h, :w])

        class Group():

            composite_keys = {'color', 'min', 'max'}
            stitch_keys = {'position', 'subregion'}

            if order == 'before':
                size = tuple(tile_size) + (3,)
                dtype = staticmethod(lambda t: np.float32)
                index = staticmethod(lambda t: tuple(t['indices']))
                first_call = staticmethod(composite)
                second_call = staticmethod(stitch)
                in_keys = composite_keys
                out_keys = stitch_keys

            if order == 'after':
                size = tuple(crop_size)
                dtype = staticmethod(lambda t: t['image'].dtype)
                index = staticmethod(lambda t: t['channel'])
                first_call = staticmethod(stitch)
                second_call = staticmethod(composite)
                in_keys = stitch_keys
                out_keys = composite_keys

            def __init__(self, t):
                d = self.dtype(t)
                self.buffer = {k: t[k] for k in self.out_keys}
                self.buffer['image'] = np.zeros(self.size, dtype=d)
                self.inputs = []
                self += t

            def __iadd__(self, t):
                self.inputs += [
                    {k: t[k] for k in self.in_keys | {'image'}}
                ]
                return self

        def hash_groups(groups, tile):
            '''
            If before: group channels by tile
            If after: group tiles by channel
            '''

            idx = Group.index(tile)

            if idx not in groups:
                groups[idx] = Group(tile)
            else:
                groups[idx] += tile

            return groups

        def combine_groups(out, group):
            '''
            If before: Composite to RGBA float tile then stitch
            If after: Stitch to gray integer image then composite
            '''
            for t in group.inputs:
                group.first_call(group.buffer['image'], t)
            group.second_call(out, group.buffer)

            return out

        inputs = [t for t in tiles if t]
        out = np.zeros(tuple(crop_size) + (3,))

        # Make groups by channel or by tile
        groups = reduce(hash_groups, inputs, {}).values()
        # Stitch and Composite in either order
        out = reduce(combine_groups, groups, out)

        # Return gamma correct image within 0, 1
        np.clip(out, 0, 1, out=out)
        return skimage.exposure.adjust_gamma(out, 1 / 2.2)

    @staticmethod
    def stitch_tiles_at_level(channels, tile_size, full_size,
                              level, order='before'):
        ''' Position all image tiles for all channels

        Args:
            tiles: Iterator of tiles to blend. Each dict in the
                list must have the following rendering settings:
                {
                    channel: Integer channel index
                    indices: Integer i, j tile indices
                    image: Numpy 2D image data of any type
                    color: Color as r, g, b float array within 0, 1
                    min: Threshhold range minimum, float within 0, 1
                    max: Threshhold range maximum, float within 0, 1
                    subregion: The start uv, end uv relative to tile
                    position: The xy position relative to origin
                }
            tile_size: width, height of one tile
            full_size: full-resolution width, height to select
            level: integer level of detail
            order: Composite `'before'` or `'after'` stitching

        Returns:
            For a given `shape` of `(width, height)`,
            returns a float32 RGB color image with shape
            `(height, width, 3)` and values in the range 0 to 1
        '''

        crop_size = crop.apply_lod(full_size, level)

        return crop.stitch_tiles(channels, tile_size, crop_size, order)

    @staticmethod
    def iterate_tiles(channels, tile_size, origin, crop_size):
        ''' Return crop settings for channel tiles

        Args:
            channels: An iterator of dicts for channels to blend. Each
                dict in the list must have the following settings:
                {
                    channel: Integer channel index
                    color: Color as r, g, b float array within 0, 1
                    min: Threshhold range minimum, float within 0, 1
                    max: Threshhold range maximum, float within 0, 1
                }
            tile_size: width, height of one tile
            origin: x, y coordinates to begin subregion
            crop_size: width, height to select

        Returns:
            An iterator of tiles to render for the given region.
            Each dict in the list has the following settings:
            {
                channel: Integer channel index
                indices: Integer i, j tile indices
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        '''

        for channel in channels:

            (r, g, b) = channel['color']
            _id = channel['channel']
            _min = channel['min']
            _max = channel['max']

            for indices in crop.select_tiles(tile_size, origin, crop_size):

                (i, j) = indices
                (x0, y0) = crop.get_position(indices, tile_size, origin)
                (u0, v0), (u1, v1) = crop.get_subregion(indices, tile_size,
                                                        origin, crop_size)

                yield {
                    'channel': _id,
                    'indices': (i, j),
                    'position': (x0, y0),
                    'subregion': ((u0, v0), (u1, v1)),
                    'color': (r, g, b),
                    'min': _min,
                    'max': _max,
                }

    @staticmethod
    def list_tiles_at_level(channels, tile_size,
                            full_origin, full_size, level):
        ''' Return crop settings all tiles at given level

        Args:
            channels: An iterator of dicts for channels to blend. Each
                dict in the list must have the following settings:
                {
                    channel: Integer channel index
                    color: Color as r, g, b float array within 0, 1
                    min: Threshhold range minimum, float within 0, 1
                    max: Threshhold range maximum, float within 0, 1
                }
            tile_size: width, height of one tile
            full_origin: full-resolution x, y coordinates to begin subregion
            full_size: full-resolution width, height to select
            level: integer level of detail

        Returns:
            An iterator of tiles to render for the given region.
            Each dict in the list has the following settings:
            {
                level: given level of detail
                channel: Integer channel index
                indices: Integer i, j tile indices
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        '''

        origin = crop.apply_lod(full_origin, level)
        crop_size = crop.apply_lod(full_size, level)

        tiles = crop.iterate_tiles(channels, tile_size, origin, crop_size)

        return [{**t, 'level': level} for t in tiles]


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


def do_crop(load_tile, channels, tile_size, origin, shape,
            levels=1, max_size=2000, order='before'):
    ''' Interface with minerva_lib.crop

    Args:
        load_tile: Function to supply 2D numpy array
        channels: List of dicts of channel rendering settings
        tile_size: The width, height of a single tile
        origin: Request's full-resolution x, y origin
        shape: Request's full-resolution width, height
        levels: The number of levels of detail
        max_size: The maximum response width or height
        order: Composite 'before' or 'after' stitching

    Returns:
        2D numpy float array of with width, height given by
        `shape` if `shape <= max_size` or width, height given by
        `shape / 2 ** l <= max_size` for the lowest `l` meeting
        `l < levels`. The array is a composite of all channels
        for full or partial tiles within `shape` from `origin`.
    '''

    # Compute parameters following figure API
    k_lod = crop.get_lod(levels, max_size, *shape)
    k_origin = crop.apply_lod(origin, k_lod)
    k_shape = crop.apply_lod(shape, k_lod)

    msg = '''Cropping 1/{0} scale:
    {2} pixels starting at {1}
    '''.format(2**k_lod, k_origin, k_shape)
    print(msg)

    def store_tile(tile):
        ''' Load an 'image' for the tile

        Args:
            A tile dict from `iterate_tiles`

        Returns:
            A tile dict for `stitch_tiles`
        '''

        c = tile['channel']
        i, j = tile['indices']

        # Disallow negative tiles
        if i < 0 or j < 0:
            return None

        # Load image from Omero
        image = load_tile(c, k_lod, i, j)

        # Disallow empty images
        if image is None:
            return None

        # Return image to Minerva
        tile['image'] = image
        return tile

    # Load and stitch all tiles
    tiles = crop.iterate_tiles(channels, tile_size,
                               k_origin, k_shape)
    return crop.stitch_tiles(map(store_tile, tiles),
                             tile_size, k_shape, order)


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
    default_url += '&region=-100,-100,1300,1300'
    cmd.add_argument(
        'url', nargs='?', default=default_url,
        help='OMERO.figure render_scaled_region url'
    )
    cmd.add_argument(
        '--after', action='store_true',
        help='composite after stitching'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help='output directory'
    )

    parsed = cmd.parse_args(args)
    out_file = str(pathlib.Path(parsed.o, 'out.png'))
    order = ['before', 'after'][parsed.after]

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
                  keys['max_size'], order)

    # Write the image buffer to a file
    try:
        skimage.io.imsave(out_file, np.uint8(255 * out))
    except OSError as o_e:
        print(o_e)


if __name__ == "__main__":
    main(sys.argv[1:])
