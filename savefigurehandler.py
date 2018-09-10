import json
import boto3
import string
import random
from mimetypes import types_map
from tornado import web


class SaveFigureHandler(web.RequestHandler):
    ''' Returns metadata
    '''
    _basic_mime = 'text/plain'
    charset = string.ascii_uppercase + string.digits
    s3 = boto3.resource('s3')

    def initialize(self, bucket):
        ''' Create new handler for metadata

        Arguments:
            bucket: s3 tile bucket name
        '''
        self.bucket = bucket

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST')

    def post(self, path):
        ''' Get image data

        Arguments:
            path: the imgData request
        '''
        data = self.parse(path)

        # Get the mimetype from the requested extension
        mime_type = types_map.get('json', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        self.write(str(data))

    def parse(self, path):
        ''' Get image data for uuid

        Arguments:
            path: '/'

        Returns:
            the imagedata dictionary
        '''

        key = ''.join(random.choices(self.charset, k=64))
        obj = self.s3.Object(self.bucket, key + '.json')

        obj.put(Body=self.request.body)
        return key
