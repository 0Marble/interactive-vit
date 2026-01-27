
import importlib
from django.conf import settings
import os
from typing import Dict
from urllib.parse import urlencode
import logging
from main.graph import Graph, Pinout
import sys

logger = logging.getLogger(__name__)

class NodeKind:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def contents(self, params: Dict[str, str]) -> str:
        return self.name + "?" + urlencode(params)

    def io(self, params: Dict[str, str]) -> Dict:
        _ = params
        raise Exception(f"TODO: implement Node.io() for {self.name}")

    def compute(self, params: Dict[str, str], inputs: Pinout) -> Pinout:
        _ = params
        _ = inputs
        raise Exception(f"TODO: implement Node.compute() for {self.name}")

class Context:
    def __init__(self):
        self.nodes: Dict[str, NodeKind] = {}

    def register(self, node: NodeKind):
        logger.info("Registered node: '%s'", node.get_name())
        self.nodes[node.get_name()] = node

    def get_node(self, name: str) -> NodeKind:
        return self.nodes[name]

    def compute(self, graph: Graph):
        for n in graph.order():
            node = self.get_node(n.name)
            pinout = node.compute(n.params, n.get_pinin())
            n.set_pinout(pinout)

instance = Context()

def context() -> Context:
    return instance

def scan_nodes():
    nodes_dir = os.path.join(settings.BASE_DIR, "main/nodes")

    for file in os.listdir(nodes_dir):
        path = os.path.join(nodes_dir, file)
        if not os.path.isfile(path) or not path.endswith(".py"): continue
        _, file_name = os.path.split(path)
        name, _ = os.path.splitext(file_name)

        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)

            for node in module.nodes(): 
                context().register(node)

        except Exception as err:
            logger.info("Could not register node at '%s': %s", path, str(err))

scan_nodes()
