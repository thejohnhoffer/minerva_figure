import png
import numpy as np
import io
from tornado import web
from mimetypes import types_map
from concurrent.futures import ThreadPoolExecutor

from minerva_lib import crop
from minerva_scripts.crop import do_crop
from minerva_scripts.omeroapi import OmeroApi
from minerva_scripts.minervaapi import MinervaApi


class RegionHandler(web.RequestHandler):
    ''' Returns static files
    '''
    _basic_mime = 'text/plain'

    def initialize(self, bucket, domain):
        ''' Create new handler for image requests

        Arguments:
            bucket: s3 tile bucket name
            domain: *.*.*.amazonaws.com/*
        '''
        self.bucket = bucket
        self.domain = domain

        self._ex = ThreadPoolExecutor(max_workers=10)
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET')

    def get(self, path):
        ''' Get image

        Arguments:
            path: the render_scaled_region or render_image request
        '''
        data = self.parse(path)

        if data is None:
            self.set_header('Content-Type', self._basic_mime)
            self.set_status(403)
            self.write('403')

        # Get the mimetype from the requested extension
        mime_type = types_map.get('png', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        with io.BytesIO() as out_file:
            png.from_array(data, mode='RGB').save(out_file)
            out_file.seek(0)
            test_bytes = out_file.read()
            self.write(test_bytes)

    def parse(self, path):
        ''' Get image for uuid

        Arguments:
            path:: render_scaled_region api

        Returns:
            the image
        '''
        split_path = path.split('/')
        token = split_path[0]
        uuid = split_path[1]

        query_args = self.request.arguments
        query_dict = {k: v[0].decode("utf-8") for k, v in query_args.items()}

        # Read parameters from URL and API
        keys = OmeroApi.scaled_region(split_path[1:], query_dict, token,
                                      self.bucket, self.domain)

        if not keys:
            return None

        # Make array of channel parameters
        inputs = zip(keys['chan'], keys['c'], keys['r'])
        channels = map(MinervaApi.format_input, inputs)

        # Minerva loads the tiles
        def ask_minerva(c, l, y, x):
            keywords = {
                't': 0,
                'z': 0,
                'l': l,
                'x': x,
                'y': y
            }
            limit = keys['limit']
            return MinervaApi.image(uuid, token, c, limit, **keywords)

        # Region with margins
        outer_origin = keys['origin']
        outer_shape = keys['shape']
        outer_end = np.array(outer_origin) + outer_shape

        # Actual image content
        image_shape = keys['image_shape']
        request_origin = np.maximum(outer_origin, 0)
        request_end = np.minimum(outer_end, image_shape)
        request_shape = request_end - request_origin

        # Return nothing for an ivalid request
        valid = crop.validate_region_bounds(request_origin, request_shape,
                                            image_shape)
        if not valid:
            return None

        # Minerva does the cropping
        image = do_crop(ask_minerva, channels, keys['tile_shape'],
                        request_origin, request_shape, keys['levels'],
                        keys['max_size'])

        # Use y, x, color output shape
        out_h, out_w = outer_shape.astype(np.int64)
        out = np.ones((out_h, out_w, 3)) * 0.5

        # Position cropped region within margins
        y_0, x_0 = (request_origin - outer_origin).astype(np.int64)
        y_1, x_1 = [y_0, x_0] + request_shape
        out[y_0:y_1, x_0:x_1] = image

        return np.uint8(255 * out)
