import json
from mimetypes import types_map
from concurrent.futures import ThreadPoolExecutor
from tornado import web, gen

from gists.minervaapi import MinervaApi


class MetaHandler(web.RequestHandler):
    ''' Returns metadata
    '''
    _basic_mime = 'text/plain'

    def initialize(self, bucket, domain):
        ''' Create new handler for metadata

        Arguments:
            bucket: s3 tile bucket name
            domain: *.*.*.amazonaws.com/*
        '''
        self.bucket = bucket
        self.domain = domain

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
            data: the imgdata dictionary
        '''
        # Get the mimetype from the requested extension
        mime_type = types_map.get('json', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        self.write(json.dumps(data))

    def parse(self, path):
        ''' Get image data for uuid

        Arguments:
            path: 'token/uuid'

        Returns:
            the imagedata dictionary
        '''

        token, uuid = path.split('/')[:2]

        bucket = self.bucket
        domain = self.domain

        return MinervaApi.load_config(uuid, token, bucket, domain)
