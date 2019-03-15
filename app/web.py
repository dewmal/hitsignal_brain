from sanic.response import html

import settings
from application import web_app


@web_app.route('/')
def index(request):
    with open('ui/template/index.html') as f:
        return html(f.read())


web_app.static('static', settings.WEB_APP_STATIC_FOLDER)
