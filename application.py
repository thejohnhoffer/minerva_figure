import tornado.web
import tornado.wsgi
import tornado.ioloop

import asyncio
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


webApp = tornado.web.Application([
    (r"/", MainHandler),
])

# Wrapping the Tornado Application into a WSGI interface
# As per AWS EB requirements, the WSGI interface must be named
# 'application' only
application = tornado.wsgi.WSGIAdapter(webApp)

if __name__ == '__main__':
    # If testing the server locally, start on the specific port
    webApp.listen(8080)
    tornado.ioloop.IOLoop.current().start()
