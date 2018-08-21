import os
import posixpath
from tornado import web, gen
from mimetypes import types_map
from pkg_resources import resource_string
from concurrent.futures import ThreadPoolExecutor


class StaticHandler(web.RequestHandler):
    ''' Returns static files
    '''
    _basic_mime = 'text/plain'

    def initialize(self, _root):
        ''' Create new handler for static data

        Arguments:
            _root: A module in the directory containing the static path

        '''
        self._root = _root
        self._ex = ThreadPoolExecutor(max_workers=10)
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET')

    @gen.coroutine
    def get(self, path):
        ''' Asynchronously call handle

        Arguments:
            path: the static path requested in the URL
        '''
        filepath = self.parse(path)
        yield self._ex.submit(self.handle, filepath)

    def handle(self, filepath):
        ''' Serves a path in the root directory

        Arguments:
            path: the actual path to a file on the server
        '''
        extension = os.path.splitext(filepath)[1]
        # Get the mimetype from the requested extension
        mime_type = types_map.get(extension, self._basic_mime)
        self.set_header('Content-Type', mime_type)

        data = resource_string(self._root, filepath)
        self.write(data)

    def parse(self, path):
        ''' Convert the requested path into a real system path

        Arguments:
            path: the static path requested in the URL

        Returns:
            the actual path to a file on the server
        '''
        if not path:
            path = ''

        index_html = 'index.html'
        # Turn directory to index
        if '.' not in os.path.basename(path):
            path = os.path.join(path, index_html)

        # Get the actual path on server
        path = posixpath.normpath(path)
        filepath = os.path.join('static', path)
        # Deny access to any path outside static directory
        if os.path.isabs(path) or path.startswith('..'):
            return self.send_error(403)
        # Return the actual path on the server
        return filepath
