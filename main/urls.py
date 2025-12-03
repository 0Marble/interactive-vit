from django.urls import path as django_path
from django.conf import settings

from . import views

import importlib
import os
import logging
import sys

logger = logging.getLogger(__name__)

urlpatterns = [
    django_path("", views.index, name="index"),
    django_path("list_graphs", views.list_graphs, name="list-graphs"),
    django_path("load_graph", views.load_graph, name="load-graph"),
    django_path("node/<str:model_name>/<str:node_name>/compute", views.model_compute),
    django_path("node/<str:model_name>/<str:node_name>/description", views.model_description),
    django_path("node/<str:model_name>/<str:node_name>/contents", views.model_contents),
]

def scan_nodes():
    nodes_dir = os.path.join(settings.BASE_DIR, "main/nodes")

    for file in os.listdir(nodes_dir):
        path = os.path.join(nodes_dir, file)
        if not os.path.isfile(path) or not path.endswith(".py"): continue
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
            urlpatterns.append(django_path(f"{endpoint}/description", mod.description, name=f"node-{name}-description"))
            urlpatterns.append(django_path(f"{endpoint}/contents", mod.contents, name=f"node-{name}-contents"))
            urlpatterns.append(django_path(f"{endpoint}/compute", mod.compute, name=f"node-{name}-compute"))

            logger.error("Registered node: '%s' -> %s", name, endpoint)
        except Exception as err:
            logger.error("Could not register node '%s': %s", path, str(err))

scan_nodes()
