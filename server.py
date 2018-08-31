''' A simple static filie and API server
'''
import sys

from tornado.ioloop import IOLoop
from tornado.web import Application
from statichandler import StaticHandler
from regionhandler import RegionHandler
from metahandler import MetaHandler

import asyncio
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


class Webserver(object):
    ''' A simple tornado webserver
    '''

    _port = 8080

    def __init__(self):

        minerva_bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
        minerva_domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'

        keys = {
            'bucket': minerva_bucket,
            'domain': minerva_domain
        }

        self._webapp = Application([
            (r'/webgateway/render_scaled_region/(.*)', RegionHandler, keys),
            (r'/webgateway/render_image/(.*)', RegionHandler, keys),
            (r'/figure/imgData/(.*)/', MetaHandler, keys),
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
        ], autoreload=False)

    def start(self):
        ''' Starts the webapp on the given port

        Arguments:
            _port: The port number to serve all entry points

        Returns:
            tornado.IOLoop needed to stop the app

        '''
        # Begin to serve the web application
        self._webapp.listen(self._port)
        # Send the logging message
        msg = '''
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
 Start server on port {0}.
_______________________________
        '''
        print(msg.format(self._port))
        return IOLoop.current()

    def stop(self):
        ''' Stops the server
        '''

        # Ask tornado to stop
        ioloop = IOLoop.current()
        ioloop.add_callback(ioloop.stop)
        # Send the stop message
        msg = '''
|||||||||||||||||||||||||||||||
 Stop server on port {0}.
*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
        '''
        print(msg.format(self._port))


def main():
    try:
        server = Webserver()
    except KeyError:
        sys.exit()
    try:
        ioloop = server.start()
        ioloop.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
