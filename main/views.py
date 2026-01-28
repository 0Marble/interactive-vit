import django.http as http
from django.template import loader
from django.conf import settings
from django.views.static import serve

import logging
import os
from main.message import Request, Response
from main.context import context

logger = logging.getLogger(__name__)

def index(request):
    template = loader.get_template("main/index.html")
    return http.HttpResponse(template.render({}, request))

def description(http_req: http.HttpRequest, name: str) -> http.HttpResponse:
    try: 
        json_obj = context().get_node(name).io(http_req.GET)
        return http.JsonResponse(json_obj, safe=False)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e).encode())

def contents(http_req: http.HttpRequest, name: str) -> http.HttpResponse:
    try: 
        return http.HttpResponse(context().get_node(name).contents(http_req.GET).encode())
    except Exception as e:
        return http.HttpResponseBadRequest(str(e).encode())

def compute(http_req: http.HttpRequest):
    try:
        req = Request()
        req.decode(http_req.body)
        logger.debug("%s", req.graph.__str__())
        context().compute(req.graph)
        logger.debug("%s", req.graph.__str__())

        resp = Response(req.graph)
        return http.HttpResponse(resp.encode())
    except Exception as e:
        logger.error(e)
        return http.HttpResponseBadRequest(str(e).encode())

def list_graphs(req: http.HttpRequest) -> http.HttpResponse:
    _ = req
    graphs_dir = os.path.join(settings.BASE_DIR, "static/graphs")
    res = []
    for model_name in os.listdir(graphs_dir):
        res.append(model_name)
    return http.JsonResponse(res, safe=False)

def load_graph(req: http.HttpRequest, name: str) -> http.HttpResponse | http.FileResponse:
    try:
        graphs_dir = os.path.join(settings.BASE_DIR, "static/graphs/")
        file = serve(req, name, document_root=graphs_dir)
        return file
    except Exception as e:
        logger.error(e)
        return http.HttpResponseBadRequest(str(e).encode())
