import time
import boto3
import string
import random
from mimetypes import types_map
from tornado import escape
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

        def make_figure_key(k):
            return f'figureJSON/{k}.json'

        def make_index_key(k):
            return f'indexJSON/{k}.json'

        uuid = ''.join(random.choices(self.charset, k=64))

        # Write the figure json to s3
        figure_obj = self.s3.Object(self.bucket, make_figure_key(uuid))
        figure_body = str(self.request.body).split('figureJSON=')[1][:-1]
        figure_body = escape.url_unescape(figure_body)
        figure_obj.put(Body=figure_body)

        # Read the figure name and first image uuid
        figure_object = escape.json_decode(figure_body)
        first_panel = next(iter(figure_object.get('panels', [])), {})
        figure_name = figure_object.get('figureName', '')
        first_image = first_panel.get('imageId', '')

        # Write the index json to s3
        index_obj = self.s3.Object(self.bucket, make_index_key(uuid))
        index_body = escape.json_encode({
            'description': escape.json_encode({
                'name': figure_name,
                'imageId': first_image
            }),
            'ownerFullName': 'Minerva',
            'creationDate': int(time.time()),
            'name': figure_name,
            'canEdit': False,
            'id': uuid,
        })
        index_obj.put(Body=index_body)

        return uuid
