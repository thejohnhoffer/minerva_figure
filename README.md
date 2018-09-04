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


rm -rf static
mv omero.hms.harvard.edu/static static
mv omero.hms.harvard.edu/figure/* static/figure/
for i in `find static -type f`; do mv $i `echo $i | cut -d? -f1`; done
```
