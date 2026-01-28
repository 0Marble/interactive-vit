from __future__ import annotations
import importlib
import json
from django.conf import settings
import os
from typing import Dict
from urllib.parse import urlencode
import logging
from main.graph import Graph, Pinout
import sys
import torch
import math

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

    def register(self, ctx: Context):
        ctx.register(self)

class Model:
    def __init__(self, model: torch.nn.Module, name: str):
        self.model = model
        self.model.eval()
        self.name = name

        self.node_names: list[str] = []
        for (name, sub) in self.model.named_modules():
            if sum([1 for _ in sub.named_modules()]) != 1: continue
            self.node_names.append(self.prefix() + name)

    def get_name(self) -> str:
        return self.name

    def prefix(self) -> str:
        return self.name + ":"

    def generate_graph_json(self) -> Dict:
        json_obj = {"nodes": [], "edges": []};
        nodes = self.list_node_names()
        cnt = len(nodes)
        w = int(math.sqrt(cnt))

        for i, name in enumerate(nodes):
            json_obj["nodes"].append({
                "instance":{"kind":"net_node","endpoint":f"{name}","params":{}},
                "pos":{"x": (i % w) * 200,"y": int(i / w) * 200}
            })

            if i != 0: 
                json_obj["edges"].append({
                    "in_port": {"node": i - 1, "channel": "o"},
                    "out_port": {"node": i, "channel": "o"}
                });

        return json_obj


    def list_node_names(self) -> list[str]:
        return self.node_names

    def compute(self, node_name: str, pinin: Pinout) -> Pinout:
        with torch.no_grad():
            sub = self.model.get_submodule(node_name.removeprefix(self.prefix()))
            x = pinin.get("o")
            assert x is not None
            res = sub(x)
            assert isinstance(res, torch.Tensor)
            out = Pinout()
            out.set("o", res)
            return out

    def contents(self, node_name: str) -> str:
        sub = self.model.get_submodule(node_name.removeprefix(self.prefix()))
        return f"<p>{node_name}</p> <p>{sub._get_name()}</p>"

    def io(self, node_name: str) -> Dict:
        _ = node_name
        return {"ins": ["o"], "outs": ["o"]}

    def register(self, ctx: Context):
        graph_dir = os.path.join(settings.BASE_DIR, "static/graphs/" + self.name + ".json")
        if not os.path.exists(graph_dir):
            try:
                json_obj = self.generate_graph_json()
                f = open(graph_dir, "w")
                f.write(json.dumps(json_obj))
                f.close()
                logger.info("generated graph %s", graph_dir)
            except Exception as e: 
                logger.error("could not generate graph %s: %s", graph_dir, str(e))

        for node_name in self.list_node_names():
            node = ModelNode(self, node_name)
            node.register(ctx)

class ModelNode(NodeKind):
    def __init__(self, parent: Model, name: str):
        super().__init__(name)
        self.parent = parent

    def compute(self, params: Dict[str, str], inputs: Pinout) -> Pinout:
        _ = params
        return self.parent.compute(self.get_name(), inputs)

    def contents(self, params: Dict[str, str]) -> str:
        _ = params
        return self.parent.contents(self.get_name())

    def io(self, params: Dict[str, str]) -> Dict:
        _ = params
        return self.parent.io(self.get_name())


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

def scan_nodes(dirs):
    for subdir in dirs:
        full_dir = os.path.join(settings.BASE_DIR, subdir)

        for file in os.listdir(full_dir):
            path = os.path.join(full_dir, file)
            if not os.path.isfile(path) or not path.endswith(".py"): continue
            _, file_name = os.path.split(path)
            name, _ = os.path.splitext(file_name)

            try:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)

                for instance in module.instances(): 
                    instance.register(context())

            except Exception as err:
                logger.info("Could not register '%s': %s", path, str(err))

scan_nodes(["main/nodes", "static/models"])
