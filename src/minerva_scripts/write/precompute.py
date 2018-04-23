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


def get_index(data_type, full_shape, block_shape, lods):
    ''' Give precomputed metadata for neuroglancer

    Arguments:
        data_type: string such as 'uint16'
        full_shape: ZYX size of full image
        block_shape: ZYX size of a single tile
        lods: the number of levels of detail

    Returns:
        dictionary expected by precomputed info API
    '''

    def index_(lod):
        ''' Provide configuration for each resolution
        '''
        res = int(2 ** lod)
        return {
            'key': str(res),
            'size': list(map(int, full_shape[::-1])),
            'resolution': [res, res, 1],
            'chunk_sizes': list(map(int, block_shape[::-1])),
            'voxel_offset': [0, 0, 0],
            'encoding': 'jpeg',
        }

    # Return configuration for all resolutions
    return {
        'num_channels': 1,
        'data_type': data_type,
        'scales': [index_(lod) for lod in range(lods)],
        'type': 'segmentation',
    }
