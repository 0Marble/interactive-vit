from django.shortcuts import render
import django.http as http
from django.template import loader
from django.conf import settings
from django.views.static import serve

import logging
import os
import importlib
import sys
import traceback
import json

import torch

from main.message import Request, Response

logger = logging.getLogger(__name__)

def index(request):
    template = loader.get_template("main/index.html")
    return http.HttpResponse(template.render({}, request))

def verify_model(model_path):
    model_graph = os.path.join(model_path, "graph.json")

    if not os.path.isdir(model_path): return False
    if not os.path.exists(model_graph): return False
    if not os.path.isfile(model_graph): return False

    return True


all_models_path = os.path.join(settings.BASE_DIR, "static/models")

def list_graphs(request: http.HttpRequest):
    res = []
    for model_name in os.listdir(all_models_path):
        model_path = os.path.join(all_models_path, model_name)
        if not verify_model(model_path): continue
        res.append(model_name)

    return http.JsonResponse(res, safe=False)

def load_graph(request: http.HttpRequest):
    try:
        assert request.method == "GET"
        model_name = request.GET.get("name")
        if model_name is None: raise Exception("no model name given")
        model_path = os.path.join(all_models_path, model_name)
        if not verify_model(model_path): raise Exception(f"{model_name}: no such model")

        file = serve(request, f"{model_name}/graph.json", document_root=all_models_path)
        return file
    except Exception as e:
        logger.error(e)
        return http.HttpResponseBadRequest(str(e).encode())


def compute(http_req: http.HttpRequest):
    req = Request()
    req.decode(http_req.body)
    resp = Response()

    for node in req.graph.order():
        x = node.inputs["o"].tensor
        assert x is not None
        y = torch.cos(x)
        if "o" in node.outputs: node.outputs["o"].tensor = y
        resp.set_output(node.index, "o", y)

    return http.HttpResponse(resp.encode())
