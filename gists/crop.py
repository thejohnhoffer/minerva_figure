''' Test to crop all tiles in a region
'''
from minerva_lib import crop


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

            # Load image from Minerva
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

if __name__ == '__main__':
    import os
    import sys
    import pathlib
    import argparse
    import skimage.io
    import numpy as np
    from aws_srp import AWSSRP
    from omeroapi import OmeroApi
    from minervaapi import MinervaApi

    args = sys.argv[1:]

    # Read from a configuration file at a default location
    cmd = argparse.ArgumentParser(
        description='Crop a region'
    )

    default_url = '/0/0/?c=1|0:65535$FF0000'
    default_url += '&region=0,0,1024,1024'
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

    # Set up AWS Authentication
    try:
        username = os.environ['MINERVA_USERNAME']
    except KeyError:
        print('must have MINERVA_USERNAME in environ', file=sys.stderr)
        sys.exit()

    try:
        password = os.environ['MINERVA_PASSWORD']
    except KeyError:
        print('must have MINERVA_PASSWORD in environ', file=sys.stderr)
        sys.exit()

    minerva_pool = 'us-east-1_YuTF9ST4J'
    minerva_client = '6ctsnjjglmtna2q5fgtrjug47k'
    uuid = '769cfb14-f583-4f22-9f48-94a24e09fd7f'
    minerva_bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
    minerva_domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'

    srp = AWSSRP(username, password, minerva_pool, minerva_client)
    result = srp.authenticate_user()
    token = result['AuthenticationResult']['IdToken']

    # Read parameters from URL and API
    split_url, query_dict = OmeroApi.read_url(uuid + parsed.url)
    keys = OmeroApi.scaled_region(split_url, query_dict, token,
                                  minerva_bucket, minerva_domain)

    # Make array of channel parameters
    inputs = zip(keys['chan'], keys['c'], keys['r'])
    channels = map(MinervaApi.format_input, inputs)

    # Minerva loads the tiles
    def ask_minerva(c, l, i, j):
        keywords = {
            't': 0,
            'z': 0,
            'l': l,
            'x': i,
            'y': j
        }
        limit = keys['limit']
        return MinervaApi.image(uuid, token, c, limit, **keywords)

    # Minerva does the cropping
    out = do_crop(ask_minerva, channels, keys['tile_size'],
                  keys['origin'], keys['shape'], keys['levels'],
                  keys['max_size'])

    # Write the image buffer to a file
    try:
        skimage.io.imsave(out_file, np.uint8(255 * out))
    except OSError as o_e:
        print(o_e)
