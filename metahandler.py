import io
import numpy as np
import skimage
import boto3
from tornado import web, gen
from mimetypes import types_map
from concurrent.futures import ThreadPoolExecutor

import gists.crop as crop


s3 = boto3.resource('s3')


class MetaHandler(web.RequestHandler):
    ''' Returns static files
    '''
    _basic_mime = 'text/plain'

    def initialize(self, token):
        ''' Create new handler for static data

        Arguments:
            token: AWS Cognito Id Token

        '''
        self.token = token
        self._ex = ThreadPoolExecutor(max_workers=10)
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET')

    @gen.coroutine
    def get(self, path):
        ''' Asynchronously call handle

        Arguments:
            path: the static path requested in the URL
        '''
        filepath = self.parse(path)
        yield self._ex.submit(self.handle, filepath)

    def handle(self, data):
        ''' Serves a path in the root directory

        Arguments:
            data: RGB image array
        '''
        # Get the mimetype from the requested extension
        mime_type = types_map.get('png', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        out_file = io.StringIO()
        skimage.io.imsave(out_file, data)
        self.write(out_file.read())

    def parse(self, path):
        ''' Get image for uuid

        Arguments:
            path:: render_scaled_region api

        Returns:
            the image
        '''
        split_path = path.split('/')
        url = ''.join(split_path[1:])
        uuid = split_path[0]
        token = self.token

        # Read parameters from URL and API
        keys = crop.api.scaled_region(url, uuid, token)

        # Make array of channel parameters
        inputs = zip(keys['chan'], keys['c'], keys['r'])
        channels = map(crop.format_input, inputs)

        # OMERO loads the tiles
        def ask_minerva(c, l, i, j):
            keywords = {
                't': 0,
                'z': 0,
                'l': l,
                'x': i,
                'y': j
            }
            limit = keys['limit']
            return crop.minerva.image(uuid, token, c, limit, **keywords)

        # Minerva does the cropping
        out = crop.do_crop(ask_minerva, channels, keys['tile_size'],
                           keys['origin'], keys['shape'], keys['levels'],
                           keys['max_size'])

        return np.uint8(255 * out)
