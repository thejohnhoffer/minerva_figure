import minerva_scripts.load.disk
import minerva_scripts.blend.mem
import numpy as np
import cv2
import sys
import os

def main(args=sys.argv):

    # Constants
    IN_DIR = "/media/john/420D-AC8E/cycif_images/40BP_59/tiles/"
    IN_NAME_FORMAT = "C{0:}-T{1:}-Z{3:}-L{2:}-Y{4:}-X{5:}.png"

    # Test parameters
    NOW = "2018_04_10/5PM"
    OUT_DIR = "/home/john/2018/data/cycif_out/{}/40BP_59/tiles/".format(NOW)
    OUT_NAME_FORMAT = "T{0:}-Z{2:}-L{1:}-Y{3:}-X{4:}.png"

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Full path format of input files
    in_path_format = os.path.join(IN_DIR, IN_NAME_FORMAT)
    out_path_format = os.path.join(OUT_DIR, OUT_NAME_FORMAT)

    # Find range of image tiles
    ctlzyx_shape, tile_shape = load.disk.index(in_path_format)
    zyx_shape = ctlzyx_shape[-3::]
    n_channel = ctlzyx_shape[0]

    ALL_THRESH = np.float32([
        [0, 0.1],
        [0, 0.05],
        [0, 0.05],
        [0, 0.05],
    ])
    ALL_BGR = np.float32([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ])
    TIME = 0
    LOD = 0

    # Process all z,y,x tiles
    for i in range(np.prod(zyx_shape)):
        z,y,x = np.unravel_index(i, zyx_shape)

        # DERP
        if z != 0:
            continue

        # from disk, load all channels for tile
        all_buffer = load.disk.tile(TIME, LOD, z, y, x, **{
            'format': in_path_format,
            'count': n_channel,
        })

        # from memory, blend all channels loaded
        img_buffer = blend.mem.tile(all_buffer, **{
            'ranges': ALL_THRESH,
            'shape': tile_shape,
            'colors': ALL_BGR,
        })

        # Write the image buffer to a file
        out_file = out_path_format.format(TIME, LOD, z, y, x)
        try:
            cv2.imwrite(out_file, img_buffer)
        except Exception as e:
            print (e)

if __name__ == "__main__":
    main(sys.argv)
