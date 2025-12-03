from django.urls import path
from django.conf import settings

from . import views

import importlib
import os
import logging
import sys

logger = logging.getLogger(__name__)

urlpatterns = [
    path("", views.index, name="index"),
    path("list_models", views.list_models, name="list-models"),
    path("load_model", views.load_model, name="load-model"),
] 

def register_node(path):
    file_dir, file_name = os.path.split(path)
    name, _ = os.path.splitext(file_name)

    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None:
            raise ImportError(f"Could not load spec from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

        endpoint = f"node/{name}"
        urlpatterns.append(path(f"{endpoint}/description", mod.description, name=f"node-{name}-description"))
        urlpatterns.append(path(f"{endpoint}/contents", mod.contents, name=f"node-{name}-contents"))
        urlpatterns.append(path(f"{endpoint}/compute", mod.compute, name=f"node-{name}-compute"))

        logger.error("Registered node: '%s' -> %s", name, endpoint)
    except Exception as err:
        logger.error("Could not register node '%s': %s", path, str(err))


nodes_dir = os.path.join(settings.BASE_DIR, "main/nodes")

for file in os.listdir(nodes_dir):
    file_path = os.path.join(nodes_dir, file)
    if not os.path.isfile(file_path) or not file_path.endswith(".py"): continue
    register_node(file_path)
