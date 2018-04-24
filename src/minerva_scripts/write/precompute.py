''' Convert to precomputed https://github.com/google/neuroglancer
    /tree/master/src/neuroglancer/datasource/precomputed
'''
import os
import numpy as np


def get_path(image, indices, block_shape, lod):
    ''' Get file path needed for precomputed file

    Arguments:
        image: np.ndarray to write as jpeg
        indices: The ZYX index of this tile
        block_shape: ZYX size of a single tile
        lod: The level of detail of this tile
    '''

    image_shape = np.uint32((1,) + image.shape[:2])
    z0, y0, x0 = indices * np.uint32(block_shape)
    z1, y1, x1 = image_shape + (z0, y0, x0)

    pattern = '{}-{}_{}-{}_{}-{}'
    file_name = pattern.format(x0, x1, y0, y1, z0, z1)
    return os.path.join(str(lod), file_name)


def get_index(data_type, lod_shapes, block_shape):
    ''' Give precomputed metadata for neuroglancer

    Arguments:
        data_type: string such as 'uint16'
        lod_shape: ZYX size of image at all LOD
        block_shape: ZYX size of a single tile

    Returns:
        dictionary expected by precomputed info API
    '''
    chunk_size = block_shape[::-1].tolist()
    all_lod = enumerate(lod_shapes)

    def index_(lod):
        ''' Provide configuration for each resolution
        '''
        l, lod_shape = lod
        res = int(2 ** l)
        return {
            'key': str(l),
            'size': lod_shape.tolist()[::-1],
            'resolution': [res, res, 1],
            'chunk_sizes': [chunk_size],
            'voxel_offset': [0, 0, 0],
            'encoding': 'jpeg',
        }

    # Return configuration for all resolutions
    return {
        'num_channels': 1,
        'data_type': data_type,
        'scales': list(map(index_, all_lod)),
        'type': 'image',
    }
