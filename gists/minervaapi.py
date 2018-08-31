''' Test to crop all tiles in a region
'''
import xml.etree.ElementTree as ET
import urllib
import json
import sys
import numpy as np
import botocore
import boto3

import skimage.io
import skimage.exposure

from .metadata_xml import parse_image

######
# Minerva API
###


class MinervaApi():

    s3 = boto3.resource('s3')

    @staticmethod
    def format_input(args):
        ''' Combine all parameters

        Args:
            id_: integer channel id
            color_: 3 r,g,b floats from 0,1
            range_: 2 min,max floats from 0,1

        Returns:
            Dictionary for minerva channel
        '''
        id_, color_, range_ = args

        return {
            'channel': id_,
            'color': color_,
            'min': range_[0],
            'max': range_[1],
        }

    @staticmethod
    def image(uuid, token, c, limit, **kwargs):
        ''' Load a single channel by pattern

        Args:
            uuid: Minerva image identifier
            token: AWS Cognito Id Token
            c: zero-based channel index
            limit: max image pixel value
            args: dict with following keys
                {x, y, z, t, level}

        Returns:
            numpy array loaded from file
        '''

        def format_channel(c):
            return f'{c},FFFFFF,0,1'

        url = 'https://lze4t3ladb.execute-api.'
        url += 'us-east-1.amazonaws.com/dev/image/'
        url += '{0}/render-tile/{x}/{y}/{z}/{t}/{l}/'.format(uuid,
                                                             **kwargs)
        url += format_channel(c)
        print(url)

        req = urllib.request.Request(url, headers={
            'Authorization': token,
            'Accept': 'image/png'
        })
        try:
            with urllib.request.urlopen(req) as f:
                return skimage.io.imread(f)[:, :, 0]
        except urllib.error.HTTPError as e:
            print(e, file=sys.stderr)
            return None

        return None

    @classmethod
    def load_config(cls, uuid, token, bucket, domain):
        '''
        Args:
            uuid: the id of image in minerva
            token: AWS Cognito Id Token
            bucket: s3 tile bucket name
            domain: *.*.*.amazonaws.com/*

        Returns:
            configuration dictionary
        '''
        metadata_file = 'metadata.xml'

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
            obj = cls.s3.Object(bucket, f'{prefix}/{metadata_file}')
            root_xml = obj.get()['Body'].read().decode('utf-8')
            root = ET.fromstring(root_xml)
            config = parse_image(root, uuid)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.", file=sys.stderr)
            return {}

        return config

    @classmethod
    def index(cls, uuid, token, bucket, domain):
        '''Find all the file paths in a range

        Args:
            uuid: the id of image in minerva
            token: AWS Cognito Id Token
            bucket: s3 tile bucket name
            domain: *.*.*.amazonaws.com/*

        Returns:
            indices: dictionary with following keys:
                limit: maximum pixel value
                levels: number of pyramid levels
                image_size: image size in x, y
                tile_size: tile size in x, y
                ctxy: integer channels, timesteps, tiles in x, y
        '''
        config = cls.load_config(uuid, token, bucket, domain)

        dtype = config['meta']['pixelsType']
        tw, th = map(config['tile_size'].get,
                     ('width', 'height'))
        w, h, c, t, z = map(config['size'].get,
                            ('width', 'height', 'c', 't', 'z'))
        y = int(np.ceil(h / th))
        x = int(np.ceil(w / tw))

        return {
            'limit': np.iinfo(getattr(np, dtype)).max,
            'levels': config['levels'],
            'image_size': [w, h],
            'tile_size': [tw, th],
            'ctxy': [c, t, x, y],
        }
