''' Create imgData result from metadata.xml
'''

from functools import reduce
import xml.etree.ElementTree as ET
import json

XSD = {
    'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
}


def make_channel(channel, keys):
    ''' Create a channel dictionary for imgData request
    '''
    chan = channel.attrib
    id_label = chan['ID']
    emission = float(chan['EmissionWavelength'])
    return {
        'label': id_label,
        'window': {
            'min': keys['min'],
            'start': keys['min'],
            'end': keys['max'],
            'max': keys['max'],
        },
        'emissionWave': emission,
        'reverseIntensity': False,
        'inverted': False,
        'coefficient': 1,
        'family': 'linear',
        'color': 'FFFFFF',
        'active': True,
    }


def factor_pairs(count):
    ''' Yield factor pairs, sorted by smallest difference
    '''
    for i in range(int(count ** 0.5), 0, -1):
        if count % i == 0:
            yield i, count // i
    yield 0, 0


def make_grid(count, width, height):
    ''' Spatially arrange images of given shape
    '''
    border = 2
    gridx, gridy = next(factor_pairs(count))

    return {
        'gridx': gridx,
        'gridy': gridy,
        'border': border,
        'width': border + gridx * (border + width),
        'height': border + gridy * (border + height),
    }


def make_scale(scales, i):
    ''' Assign i to ith division by 2
    '''
    scales[str(i)] = 1.0 / (2 ** i)
    return scales


def get_image(uuid):
    ''' Simulate database result
    '''
    return {
        'pyramid_levels': 1,
        'uuid': uuid
    }


def get_uuid(attributes):
    ''' Get the UUID from an XML attributes dictionary
    '''
    return attributes['ID'].split(':').pop()


def make_meta(props):
    ''' Make metadata from XML attributes
    '''

    img = props['Image']
    pix = props['Pixels']

    return {
        'imageId': get_uuid(img),
        'imageName': img['Name'],
        'pixelsType': pix['Type'],
        'imageAuthor': 'Minerva',
        'projectDescription': '',
        'datasetDescription': '',
        'imageDescription': '',
        'imageTimestamp': '',
        'wellSampleId': '',
        'datasetName': '',
        'projectName': '',
        'projectId': '',
        'datasetId': '',
        'wellId': '',
    }


def make_image(image, props, channels, keys):
    ''' Make imgData dictionary from attriutes
    '''
    pix = props['Pixels']
    plane = props['Plane']

    size = {
        'c': int(pix['SizeC']),
        't': int(pix['SizeT']),
        'z': int(pix['SizeZ']),
        'width': int(pix['SizeX']),
        'height': int(pix['SizeY']),
    }

    levels = int(image['pyramid_levels'])
    scales = reduce(make_scale, range(levels), {})
    return {
        'size': size,
        'levels': levels,
        'channels': channels,
        'meta': make_meta(props),
        'zoomLevelScaling': scales,
        'pixel_range': [
            keys['min'],
            keys['max']
        ],
        'pixel_size': {
            'x': float(pix['PhysicalSizeX']),
            'y': float(pix['PhysicalSizeY']),
            'z': float(pix['PhysicalSizeZ']),
        },
        'deltaT': [
            float(plane['DeltaT'])
        ],
        'split_channel': {
            'g': make_grid(size['c'], size['width'], size['height']),
            'c': make_grid(size['c'] + 1, size['width'], size['height'])
        },
        'id': image['uuid'],
        'tile_size': {
            'width': 1024,
            'height': 1024,
        },
        'init_zoom': 0,
        'tiles': True,
        'interpolate': True,
        'perms': {
            'canAnnotate': False,
            'canDelete': False,
            'canEdit': False,
            'canLink': False,
        },
        'rdefs': {
            'defaultT': 0,
            'defaultZ': 0,
            'model': 'color',
            'invertAxis': False,
            'projection': 'normal',
        },
    }


def parse_image(ome, image_uuid):
    ''' Make imgData dictionary from metadata / db
    '''
    image_id = f'@ID="Image:{image_uuid}"'
    e_image = ome.find(f'ome:Image[{image_id}]', XSD)
    if not e_image:
        return {}

    e_pixels = e_image.find('ome:Pixels', XSD)
    e_plane = e_pixels.find('ome:Plane', XSD)
    e_channels = e_pixels.findall('ome:Channel', XSD)

    image = get_image(image_uuid)
    props = {
        'Pixels': e_pixels.attrib,
        'Plane': e_plane.attrib,
        'Image': e_image.attrib,
    }
    keys = {
        'min': 0,
        'max': 2 ** int(props['Pixels']['SignificantBits'])
    }
    channels = [make_channel(c, keys) for c in e_channels]

    return make_image(image, props, channels, keys)


def main():
    ''' write imgData json file
    '''
    image_uuid = 'b9e36f16-75a3-4a11-be88-ed838c3b9141'
    metadata_file = 'metadata.xml'
    imgdata_file = 'imgdata.json'

    root = ET.parse(metadata_file).getroot()
    imgdata = parse_image(root, image_uuid)

    with open(imgdata_file, 'w') as idf:
        json.dump(imgdata, idf)


if __name__ == '__main__':
    main()
