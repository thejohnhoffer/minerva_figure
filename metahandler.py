import sys
import json
import boto3
import urllib
import botocore
from tornado import web, gen
from mimetypes import types_map
from concurrent.futures import ThreadPoolExecutor

from gists import metadata_xml
import xml.etree.ElementTree as ET


s3 = boto3.resource('s3')


class MetaHandler(web.RequestHandler):
    ''' Returns static files
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

        metadata_file = 'metadata.xml'
        bucket = self.bucket
        domain = self.domain

        url = f'https://{domain}/image/{uuid}'

        req = urllib.request.Request(url, headers={
            'Authorization': token
        })
        try:
            with urllib.request.urlopen(req) as f:
                result = json.loads(f.read())
                prefix = result['data']['fileset_uuid']

        except urllib.error.HTTPError as e:
            print(e, file=sys.stderr)
            return {}

        try:
            obj = s3.Object(bucket, f'{prefix}/{metadata_file}')
            root_xml = obj.get()['Body'].read().decode('utf-8')
            root = ET.fromstring(root_xml)
            config = metadata_xml.parse_image(root, uuid)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.", file=sys.stderr)
            return {}

        return config
