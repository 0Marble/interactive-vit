from torchvision.models import vgg16
import torch
import torchvision

import math

from main.context import Model 
from main.graph import Pinout

class VggModel(Model):
    def __init__(self):
        self.weights = torchvision.models.VGG16_Weights.DEFAULT
        vgg = vgg16(weights=self.weights)
        super().__init__(vgg, "vgg16")

    def generate_graph_json(self):
        json_obj = super().generate_graph_json()
        i = json_obj["nodes"].__len__();
        w = int(math.sqrt(i))

        json_obj["nodes"].append({
            "instance":{"kind":"category", "cats": self.weights.meta["categories"]},
            "pos":{"x": (i % w) * 200, "y": int(i / w) * 200}
        })
        json_obj["edges"].append({
            "in_port": {"node": i - 1, "channel": "o"},
            "out_port": {"node": i, "channel": "o"}
        })
        return json_obj

    def list_node_names(self):
        l = super().list_node_names()
        l.insert(0, "vgg16:transform")
        l.insert(33, "vgg16:flatten")
        return l

    def compute(self, node_name: str, pinin: Pinout) -> Pinout:
        x = pinin.get("o")
        assert x is not None
        if node_name == "vgg16:transform":
            trans = self.weights.transforms()
            y = trans(x)
        elif node_name == "vgg16:flatten":
            y = torch.flatten(x)
        else: return super().compute(node_name, pinin)

        assert isinstance(y, torch.Tensor)
        res = Pinout()
        res.set("o", y)
        return res

    def contents(self, node_name: str):
        if node_name == "vgg16:transform":
            return f"<p>{node_name}</p>"
        elif node_name == "vgg16:flatten":
            return f"<p>{node_name}</p>"
        else:
            return super().contents(node_name)


def instances() -> list[Model]:
    return [VggModel()]
