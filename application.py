''' A simple static filie and API server
'''
from tornado.wsgi import WSGIAdapter
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

    port = 8080

    def __init__(self):

        minerva_bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
        minerva_domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'

        keys = {
            'bucket': minerva_bucket,
            'domain': minerva_domain
        }

        self.webApp = Application([
            (r'/webgateway/render_scaled_region/(.*)', RegionHandler, keys),
            (r'/webgateway/render_image/(.*)', RegionHandler, keys),
            (r'/figure/imgData/(.*)/', MetaHandler, keys),
            (r'/webgateway/open_with/(.*)', StaticHandler, {
                'root': __name__,
                'index': 'index.json',
                'subfolder': 'open_with'
            }),
            (r'/static/(.*)', StaticHandler, {
                'root': __name__,
                'index': 'index.html',
                'subfolder': 'static'
            }),
            (r'/(.*)', StaticHandler, {
                'root': __name__,
                'index': 'index.html',
                'subfolder': 'static'
            })
        ], autoreload=False)


server = Webserver()
port = server.port
webApp = server.webApp
application = WSGIAdapter(webApp)

if __name__ == "__main__":

    try:
        webApp.listen(port)
        print(f'Serving on {port}')
        IOLoop.current().start()
    except KeyboardInterrupt:
        ioloop = IOLoop.current()
        print(f'Closing server on {port}')
        ioloop.add_callback(ioloop.stop)
