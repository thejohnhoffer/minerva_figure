import re
import struct
import numpy as np

from .minervaapi import MinervaApi


class OmeroApi():

    @staticmethod
    def read_url(url):

        def api_index(uri):
            pattern = '(render_image|render_scaled_region)'
            match = re.search(pattern, uri)
            return match.end() if match else 0

        url = url[api_index(url):]
        if url[0] == '/':
            url = url[1:]

        split_url = url.split('?')[0].split('/')
        query = url.split('?')[1]
        query_dict = {}

        for param in query.split('&'):
            key, value = param.split('=')
            query_dict[key] = value

        return split_url, query_dict

    @staticmethod
    def scaled_region(split_url, query_dict, token, bucket, domain):
        ''' Just parse the rendered_scaled_region API
        Arguments:
            split_url: uuid, z, t
            query_dict: {
                c: comma seperated 'index|min:max$RRGGBB'
                maps: '[{"reverse":{"enabled":false}}]'
                m: c
            }
            token: AWS Cognito Id Token
            bucket: s3 tile bucket name
            domain: *.*.*.amazonaws.com/*

        Return Keywords:
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
        '''

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

        uuid, z, t = split_url[:3]
        max_size = query_dict.get('max_size', 2000)
        region = query_dict.get('region', None)
        channels = query_dict['c'].split(',')

        # Extract channel ids from channels
        channels = list(map(parse_channel, channels))
        chans = [c for c in channels if c['shown']]

        # Make API request to interpret url
        meta = MinervaApi.index(uuid, token, bucket, domain)

        if meta is None:
            return None

        if region is not None:
            x, y, width, height = parse_region(region)
            shape = np.array([width, height])
            origin = np.array([x, y])
        else:
            shape = np.array(meta['image_size'])
            origin = np.array([0, 0])

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
            'image_size': meta['image_size'],
            'r': np.array([get_range(c) for c in chans]),
            'c': np.array([get_color(c) for c in chans]),
            'chan': np.int64([c['cid'] for c in chans]),
            'max_size': int(max_size),
            'origin': origin,
            'shape': shape,
            't': int(t),
            'z': int(z)
        }
