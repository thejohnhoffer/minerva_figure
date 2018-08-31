### Installation

```bash
cd minerva-scripts
pip install -r requirements.txt -e .
```

### Running server

```bash
serve
```

### Updating Omero

After logging in to `https://omero.hms.harvard.edu/figure/`, copy a cookie of this form:

```
_ga=GA*.*.*********.**********; _gid=GA*.*.*********.**********; csrftoken=********************************; sessionid=********************************
```

Assign the cookie to a variable and use it to download the files for OMERO.figure
```
OME_COOKIE=_ga=GA*.*.*********.**********; _gid=GA*.*.*********.**********; csrftoken=********************************; sessionid=********************************"

wget -r -p -U Mozilla https://omero.hms.harvard.edu/figure/ --header="Cookie: $OME_COOKIE" --no-check-certificate --content-disposition

mv omero.hms.harvard.edu static
for i in `find static -type f`; do mv $i `echo $i | cut -d? -f1`; done
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
export OME_COOKIE="csrftoken=<TOKEN>; sessionid=<SESSION>"
crop -y examples/crop_simple.yaml -o output/folder
```
