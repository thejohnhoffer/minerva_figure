### Installation

```bash
cd minerva-scripts
pip install -r requirements.txt -e .
```

### Testing Combine

Combine all channels for all images of the format `C{CHANNEL}-T{TIME}-Z{SECTION}-L{RESOLUTION}-Y{VERTICAL_INDEX}-X{HORIZONTAL_INDEX}.png` into images of the format `T{TIME}-Z{SECTION}-L{RESOLUTION}-Y{VERTICAL_INDEX}-X{HORIZONTAL_INDEX}.png` with the following example:

```bash
combine examples/combine_ashlar_tiles.yaml -i input/folder -o output/folder
```

### Testing Crop

Crop any region from any omero image id.
Get the `Cookie` header to any valid omero request. Export value as `OME_COOKIE`

```bash
export OME_COOKIE="csrftoken=<TOKEN>; sessionid=<SESSION:
crop examples/crop_simple.yaml -o output/folder
```
