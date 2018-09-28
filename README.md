### Installation

After logging in to `https://omero.hms.harvard.edu/figure/`, copy a cookie of this form:

```
csrftoken=********************************; sessionid=********************************
```

Assign the cookie to a variable and use it to download the files for OMERO.figure

```
OME_COOKIE="csrftoken=********************************; sessionid=********************************"

wget -r -p -U Mozilla https://omero.hms.harvard.edu/figure/ --header="Cookie: $OME_COOKIE" --no-check-certificate
mv omero.hms.harvard.edu/* .
for i in `find . -type f`; do mv $i `echo $i | cut -d? -f1`; done
vim figure/index.html '+:source test.vim | wq!'
```

For a http server in python3, run `python -m http.server`
Then open `localhost:8000/figure` in a web browser.
