import numpy as np
import urllib.request
import skimage.io
import json
import ssl
import os
import io


HEADERS = {
    'Cookie': os.environ['OME_COOKIE'],
}

ssl._create_default_https_context = ssl._create_unverified_context


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
    url += '{}/{}/{}/?m=g&format={}&tile={},{},{}'.format(*args)
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


def tile(t, l, z, y, x, c_order, image_id,
         limit=65535, img_fmt='tif'):
    '''Load all channels for a given tile
    Arguments:
        t: integer time step
        l: interger level of detail (powers of 2)
        z: tile offset in depth
        y: vertical tile offset
        x: horizontal tile offset
        c_order: list of channels to load
        image_id: the id of image in omero
        limit: max pixel value
        img_fmt: image encoding

    Returns:
        list of numpy image channels for a tile
    '''
    # Load all channels
    const = limit, image_id, z, t, img_fmt, l, x, y
    return [image(c, *const) for c in c_order]


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

    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as response:
        config = json.loads(response.read())

    lod = config['levels']
    dtype = config['meta']['pixelsType']
    tw, th = map(config['tile_size'].get,
                 ('width', 'height'))
    w, h, c, t, z = map(config['size'].get,
                        ('width', 'height', 'c', 't', 'z'))
    y = int(np.ceil(h / th))
    x = int(np.ceil(w / tw))

    return {
        'limit': np.iinfo(getattr(np, dtype)).max,
        'indices': [c, t, lod, y, x],
        'tile': [th, tw],
    }
