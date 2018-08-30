''' A simple static filie and API server
'''
import sys
import argparse

from tornado.ioloop import IOLoop
from tornado.web import Application
from statichandler import StaticHandler
from regionhandler import RegionHandler
from metahandler import MetaHandler


class Webserver(object):
    ''' A simple tornado webserver
    '''

    def __init__(self):

        minerva_bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
        minerva_domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'

        self._webapp = Application([
            (r'/webgateway/render_scaled_region/(.*)', RegionHandler, {}),
            (r'/webgateway/render_image/(.*)', RegionHandler, {}),
            (r'/figure/imgData/(.*)/', MetaHandler, {
                'bucket': minerva_bucket,
                'domain': minerva_domain
            }),
            (r'/webgateway/open_with/(.*)', StaticHandler, {
                'root': __name__,
                'index': 'index.json',
                'subfolder': 'open_with'
            }),
            (r'/(.*)', StaticHandler, {
                'root': __name__,
                'index': 'index.html',
                'subfolder': 'static'
            }),
        ], autoreload=True)

        self._port = None
        self._server = None

    def start(self, _port):
        ''' Starts the webapp on the given port

        Arguments:
            _port: The port number to serve all entry points

        Returns:
            tornado.IOLoop needed to stop the app

        '''
        self._port = _port
        # Begin to serve the web application
        self._webapp.listen(_port)
        self._server = IOLoop.instance()
        # Send the logging message
        msg = '''
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
 Start server on port {0}.
_______________________________
        '''
        print(msg.format(_port))
        # Return the webserver
        return self._server

    def stop(self):
        ''' Stops the server
        '''

        # Ask tornado to stop
        ioloop = self._server
        ioloop.add_callback(ioloop.stop)
        # Send the stop message
        msg = '''
|||||||||||||||||||||||||||||||
 Stop server on port {0}.
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
        '''
        print(msg.format(self._port))


def parse_argv(argv):
    ''' Parses command line arguments

    Arguments:
        argv: array of space-separated strings entered into shell

    Returns:
        Dictionary containing port
    '''

    parser = argparse.ArgumentParser(prog='server',
                                     description='a server')
    parser.add_argument('port', type=int, nargs='?',
                        default=8000, help='a port')

    parsed = parser.parse_args(argv)
    return vars(parsed)


def main(*argv):
    args = parse_argv(argv)
    port = args['port']

    # Start a webserver on given port
    try:
        server = Webserver()
    except KeyError:
        sys.exit()
    try:
        ioloop = server.start(port)
        ioloop.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
