''' Test to crop all tiles in a region
'''
import numpy as np
import skimage.exposure
import skimage.io
import argparse
import pathlib
import struct
import sys
import re

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


def composite_channels(channels):
    '''Render each image in _channels_ additively into a composited image

    Args:
        channels: List of dicts for channels to blend. Each dict in the
            list must have the following rendering settings:
            {
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }

    Returns:
        For input images with shape `(n,m)`,
        returns a float32 RGB color image with shape
        `(n,m,3)` and values in the range 0 to 1
    '''

    num_channels = len(channels)

    # Must be at least one channel
    if num_channels < 1:
        raise ValueError('At least one channel must be specified')

    # Ensure that dimensions of all channels are equal
    shape = channels[0]['image'].shape
    for channel in channels:
        if channel['image'].shape != shape:
            raise ValueError('All channel images must have equal dimensions')

    # Shape of 3 color image
    shape_color = shape + (3,)

    # Final buffer for blending
    out_buffer = np.zeros(shape_color, dtype=np.float32)

    # rescaled images and normalized colors
    for channel in channels:

        # Add all three channels to output buffer
        args = map(channel.get, ['image', 'color', 'min', 'max'])
        composite_channel(out_buffer, *args, out=out_buffer)

    # Return gamma correct image within 0, 1
    np.clip(out_buffer, 0, 1, out=out_buffer)
    return skimage.exposure.adjust_gamma(out_buffer, 1 / 2.2)


######
# crop.py
#
# Minerva Library PR #21
# https://github.com/sorgerlab/minerva-lib-python/pull/21/files
#
# Pip requirement
# git+https://github.com/thejohnhoffer/minerva-lib-python@crop#minerva-lib
###


def get_lod(lods, max_size, width, height):
    ''' Calculate the level of detail

    Arguments:
        lods: Number of available levels of detail
        max_size: Maximum image extent in x or y
        width: Extent of image in x
        height: Extent of image in y

    Returns:
        Integer power of 2 level of detail
    '''

    longest_side = max(width, height)
    lod = np.ceil(np.log2(longest_side / max_size))
    return int(np.clip(lod, 0, lods - 1))


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


def select_tiles(tile_size, origin, crop_size):
    ''' Select tile coordinates covering crop region

    Args:
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        List of integer i, j tile indices
    '''
    start = np.array(origin)
    end = start + crop_size
    fractional_start = start / tile_size
    fractional_end = end / tile_size

    # Round to get indices containing selection
    first_index = np.int64(np.floor(fractional_start))
    last_index = np.int64(np.ceil(fractional_end))

    # Calculate all indices between first and last
    index_shape = last_index - first_index
    offsets = np.argwhere(np.ones(index_shape))
    indices = first_index + offsets

    return indices.tolist()


def get_tile_bounds(indices, tile_size, origin, crop_size):
    ''' Define subregion to extract relative to tile

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        start uv, end uv relative to tile
    '''

    crop_end = np.int64(origin) + crop_size
    tile_start = np.int64(indices) * tile_size
    tile_end = tile_start + tile_size

    # Relative to tile start
    return [
        np.maximum(origin, tile_start) - tile_start,
        np.minimum(tile_end, crop_end) - tile_start
    ]


def get_out_bounds(indices, tile_size, origin, crop_size):
    ''' Define position of cropped tile relative to origin

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        start xy, end xy relative to origin
    '''

    crop_end = np.int64(origin) + crop_size
    tile_start = np.int64(indices) * tile_size
    tile_end = tile_start + tile_size

    # Relative to origin
    return [
        np.maximum(origin, tile_start) - origin,
        np.minimum(tile_end, crop_end) - origin
    ]


def stitch_channels(out, tile_bounds, out_bounds, channels):
    ''' Position channels from tile into output image

    Args:
        out: 2D numpy array to contain stitched channels
        tile_bounds: start uv, end uv to get from tile
        out_bounds: start xy, end xy to put in _out_
        channels: List of dicts for channels to blend.
            Each dict in the list must have the
            following rendering settings:
            {
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }

    Returns:
        A reference to the modified _out_ array
    '''

    # Take data from tile
    composite = composite_channels(channels)
    [u0, v0], [u1, v1] = tile_bounds
    subregion = composite[v0:v1, u0:u1]

    # Adjust bounds to actual subregion shape
    x0, y0 = out_bounds[0]
    x1, y1 = out_bounds[0] + subregion.shape[:2][::-1]

    # Assign subregion to output
    out[y0:y1, x0:x1] = subregion

    return out


######
# Minerva Library usage
###


def get_tile(c_order, range_limit, tile_shape, *args):
    ''' Fake tile_loading with test images
    '''
    def get_channel(c, l, x, y):
        out = np.zeros(tile_shape)
        rectangle = tile_shape / 2 ** (np.array([x, y]) + c)
        y0, x0 = np.int64((tile_shape - rectangle) / 2)
        y1, x1 = np.int64((tile_shape + rectangle) / 2)
        out[y0:y1, x0:x1] = range_limit / 2 ** l
        return out

    return [get_channel(c, *args) for c in c_order]


def format_input(args):
    ''' Combine all parameters
    '''
    image_, color_, range_ = args
    if image_ is None:
        return None
    return {
        'image': image_,
        'color': color_,
        'min': range_[0],
        'max': range_[1],
    }


def parse_api(url, range_limit):
    """ Just parse the rendered_scaled_region API
    Arguments:
        url: "<matching OMERO.figure API>"
        range_limit: the maximum integer for ranges

    Return Keywords:
        max_size: maximum extent in x or y
        origin:
            integer [x, y]
        shape:
            [width, height]
        chan: integer N channels by 1 index
        l: integer power-of-2 level-of-detail
        r: float32 N channels by 2 min, max
        c: float32 N channels by 3 r, g, b
    """

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

    # Ignore .../<image_id>/<z>/<t>/?
    query = url.split('?')[-1]
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

    def get_range(chan):
        r = np.array([chan['min'], chan['max']])
        return np.clip(r / range_limit, 0, 1)

    def get_color(chan):
        c = np.array(chan['color']) / 255
        return np.clip(c, 0, 1)

    return {
        'r': np.array([get_range(c) for c in chans]),
        'c': np.array([get_color(c) for c in chans]),
        'chan': np.int64([c['cid'] for c in chans]),
        'max_size': int(max_size),
        'origin': origin,
        'shape': shape
    }


def main(args):
    ''' Crop a region
    '''
    GRAY = 221
    LIMIT = 65535
    N_LEVELS = 4
    TILE_SHAPE = np.int64([512, 512])

    # API for rendering channels 0 and 2
    URL = '/render_scaled_region/ignore/ignore/ignore/'
    URL += '?c=1|0:65535$FF0000,3|0:65535$0000FF'
    URL += '&region=-100,-100,1300,1300'

    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description='Crop a region'
    )
    cmd.add_argument(
        '-o', default=str(pathlib.Path.cwd()),
        help='output directory'
    )

    parsed = vars(cmd.parse_args(args))
    out_file = str(pathlib.Path(parsed['o']).joinpath('out.png'))
    # Actually parse and read arguments
    terms = parse_api(URL, LIMIT)

    # Parameters from config url
    all_ranges = terms['r']
    all_colors = terms['c']
    channel_order = terms['chan']
    max_size = terms['max_size']
    k_w, k_h = terms['shape']

    # Compute parameters following figure API
    k_detail = get_lod(N_LEVELS, max_size, k_w, k_h)
    k_origin = apply_lod(terms['origin'], k_detail)
    k_shape = apply_lod(terms['shape'], k_detail)

    # Create variables for cropping
    out = (GRAY / 255) * np.ones(tuple(k_shape) + (3,))
    metrics = TILE_SHAPE, k_origin, k_shape

    # stitch tiles for all tile indices
    for indices in select_tiles(*metrics):

        x, y = indices

        # Skip if indices below zero
        if x < 0 or y < 0:
            continue

        # from disk, load all channels for tile
        all_buffer = get_tile(channel_order, LIMIT, TILE_SHAPE,
                              k_detail, x, y)
        all_in = zip(all_buffer, all_colors, all_ranges)
        channels = [c for c in map(format_input, all_in) if c]

        # Calculate bounds for given indices
        tile_bounds = get_tile_bounds(indices, *metrics)
        out_bounds = get_out_bounds(indices, *metrics)
        stitch_channels(out, tile_bounds, out_bounds, channels)

    # Write the image buffer to a file
    try:
        skimage.io.imsave(out_file, np.uint8(255*out))
    except OSError as o_e:
        print(o_e)


if __name__ == '__main__':
    main(sys.argv[1:])
