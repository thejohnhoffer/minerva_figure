''' A simple static filie and API server
'''
import sys
import os
import argparse

from tornado.ioloop import IOLoop
from tornado.web import Application
from statichandler import StaticHandler
from regionhandler import RegionHandler
from metahandler import MetaHandler

from gists.aws_srp import AWSSRP


class Webserver(object):
    ''' A simple tornado webserver
    '''

    def __init__(self):

        # Set up AWS Authentication
        try:
            username = os.environ['MINERVA_USERNAME']
        except KeyError:
            print('must have MINERVA_USERNAME in environ', file=sys.stderr)
            raise

        try:
            password = os.environ['MINERVA_PASSWORD']
        except KeyError:
            print('must have MINERVA_PASSWORD in environ', file=sys.stderr)
            raise

        minerva_pool = 'us-east-1_YuTF9ST4J'
        minerva_client = '7gv29ie4pak64c63frt93mv8lq'

        srp = AWSSRP(username, password, minerva_pool, minerva_client)
        result = srp.authenticate_user()
        token = result['AuthenticationResult']['IdToken']

        app_in = {
            'token': token
        }
        stat_in = {
            '_root': __name__
        }
        # Create the webapp with both access layers
        self._webapp = Application([
            (r'/render_scaled_region/(.*)', RegionHandler, app_in),
            (r'/imgData/(.*)', MetaHandler, app_in),
            # A file requested from root of static
            (r'/()', StaticHandler, stat_in),
            (r'/(omero_figure/.*)', StaticHandler, stat_in),
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

    parsed = parser.parse_args(argv[1:])
    return vars(parsed)


if __name__ == "__main__":
    ARGS = parse_argv(sys.argv)
    PORT = ARGS['port']

    # Start a webserver on given port
    try:
        SERVER = Webserver()
    except KeyError:
        sys.exit()
    try:
        IOLOOP = SERVER.start(PORT)
        IOLOOP.start()
    except KeyboardInterrupt:
        SERVER.stop()
