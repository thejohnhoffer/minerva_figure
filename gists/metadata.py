import xml.etree.ElementTree as ET
import json

XSD = {
    'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
}


def parse_channel(channel, pix):
    chan = channel.attrib
    id_label = chan['ID']
    max_range = 2 ** int(pix['SignificantBits'])
    emission = float(chan['EmissionWavelength'])
    return {
        'label': id_label,
        'window': {
            'min': 0,
            'start': 0,
            'end': max_range,
            'max': max_range,
        },
        'emissionWave': emission,
        'reverseIntensity': False,
        'inverted': False,
        'coefficient': 1,
        'family': 'linear',
        'color': 'FFFFFF',
        'active': True,
    }


def parse_image(ome, image_uuid):
    image_id = f'@ID="Image:{image_uuid}"'
    e_image = ome.find(f'ome:Image[{image_id}]', XSD)
    if not e_image:
        return {}

    e_pixels = e_image.find('ome:Pixels', XSD)
    e_channels = e_pixels.findall('ome:Channel', XSD)

    pix = e_pixels.attrib
    channels = [parse_channel(c, pix) for c in e_channels]
    return {
        'channels': channels
    }


def main():
    image_uuid = 'b9e36f16-75a3-4a11-be88-ed838c3b9141'
    metadata_file = 'metadata.xml'
    imgdata_file = 'imgdata.json'

    root = ET.parse(metadata_file).getroot()
    imgdata = parse_image(root, image_uuid)

    with open(imgdata_file, 'w') as idf:
        json.dump(imgdata, idf)


if __name__ == '__main__':
    main()
