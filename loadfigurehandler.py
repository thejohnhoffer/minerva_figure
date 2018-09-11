import boto3
from mimetypes import types_map
from tornado import web


class LoadFigureHandler(web.RequestHandler):
    ''' Returns figure json
    '''
    _basic_mime = 'text/plain'
    s3 = boto3.resource('s3')

    def initialize(self, bucket):
        ''' Create new handler for figure json

        Arguments:
            bucket: s3 tile bucket name
        '''
        self.bucket = bucket

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET')

    def get(self, path):
        ''' Get figure data

        Arguments:
            path: figure id
        '''
        data = self.parse(path)

        # Get the mimetype from the requested extension
        mime_type = types_map.get('json', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        self.write(data)

    def parse(self, path):
        ''' Get figure json for uuid

        Arguments:
            path: figure id

        Returns:
            the json text defining the figure
        '''

        key = f'figureJSON/{path}.json'
        print(key)
        obj = self.s3.Object(self.bucket, key)
        return obj.get()['Body'].read().decode('utf-8')
