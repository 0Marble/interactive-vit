from django.shortcuts import render
import django.http as http
from django.template import loader
from django.conf import settings
from django.views.static import serve
from main.model import Model

import logging
import os
import importlib
import sys
import traceback
import json

from main.message import Message

logger = logging.getLogger(__name__)

def setup_model_objects():
    models_dir = os.path.join(settings.BASE_DIR, "static/models")
    model_objects = {}

    for model_name in os.listdir(models_dir):
        path = os.path.join(models_dir, model_name)
        model_path = os.path.join(path, "model.py")
        graph_path = os.path.join(path, "graph.json")

        if not os.path.isfile(model_path): 
            continue

        try:
            spec = importlib.util.spec_from_file_location(model_name, model_path)
            if spec is None:
                raise Exception(f"Could not create spec for model '{model_name}'")

            module = importlib.util.module_from_spec(spec)
            sys.modules[model_name] = module
            spec.loader.exec_module(module)

            model_obj: Model = module.Model()
            model_objects[model_name] = model_obj
            if not os.path.exists(graph_path):
                json_obj = model_obj.generate_graph()
                with open(graph_path, 'w') as f:
                    json.dump(json_obj, f)

            logger.error("Registered model '%s'", model_name)
        except Exception as err:
            logger.error("Could not register model '%s': %s", model_name, traceback.format_exc())

    return model_objects

models = setup_model_objects()

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
        return http.HttpResponseBadRequest(str(e))

def model_compute(request: http.HttpRequest, model_name: str, node_name: str):
    try:
        assert request.method == "POST"

        msg = Message()
        msg.decode(request.body)
        msg = models[model_name].eval_node(node_name, msg)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        logger.error(traceback.format_exc())
        return http.HttpResponseBadRequest(f"{node_name} : {str(e)}")

def model_description(req: http.HttpRequest, model_name: str, node_name: str):
    try:
        json = models[model_name].node_io_description(node_name)
        return http.JsonResponse(json, safe=False)
    except Exception as e:
        logger.error(traceback.format_exc())
        return http.HttpResponseBadRequest(str(e))

def model_contents(req: http.HttpRequest, model_name: str, node_name: str):
    try:
        html = models[model_name].node_html_contents(node_name)
        return http.HttpResponse(html)
    except Exception as e:
        logger.error(e)
        return http.HttpResponseBadRequest(str(e))

