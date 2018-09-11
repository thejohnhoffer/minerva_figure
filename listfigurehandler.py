import boto3
from mimetypes import types_map
from tornado import web


class ListFigureHandler(web.RequestHandler):
    ''' Returns metadata
    '''
    _basic_mime = 'text/plain'
    s3 = boto3.resource('s3')

    def initialize(self, bucket):
        ''' Create new handler for metadata

        Arguments:
            bucket: s3 tile bucket name
        '''
        self.bucket = bucket

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET')

    def get(self, path):
        ''' Get image data

        Arguments:
            path: the imgData request
        '''
        data = self.parse(path)

        # Get the mimetype from the requested extension
        mime_type = types_map.get('json', self._basic_mime)
        self.set_header('Content-Type', mime_type)

        self.write(data)

    def parse(self, path):
        ''' Get image data for uuid

        Arguments:
            path: '/'

        Returns:
            the imagedata dictionary
        '''

        bucket = self.s3.Bucket(self.bucket)
        index_bodies = []

        for obj in bucket.objects.all():
            if 'indexJSON/' not in obj.key:
                continue
            index_body = obj.get()['Body'].read().decode('utf-8')
            index_bodies.append(index_body)

        out = '[{}]'.format(','.join(index_bodies))
        return out
