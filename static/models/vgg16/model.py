from torchvision.models import vgg16
import torch
import torchvision

import math

import main.model
from main.message import Message

class Model(main.model.Model):
    def __init__(self):
        self.weights = torchvision.models.VGG16_Weights.DEFAULT
        v = vgg16(weights=self.weights)
        super().__init__(v, "model.")

    def name(self):
        return "vgg16"

    def generate_graph(self):
        json_obj = super().generate_graph()
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
        l.insert(0, "transform")
        l.insert(33, "flatten")
        return l

    def eval_node(self, node_name: str, in_msg: Message) -> Message:
        if node_name == "transform":
            trans = self.weights.transforms()
            res = trans(in_msg.get("o"))
        elif node_name == "flatten":
            res = torch.flatten(in_msg.get("o"))
        else: return super().eval_node(node_name, in_msg)

        assert isinstance(res, torch.Tensor)
        out_msg = Message()
        out_msg.set("o", res)
        return out_msg 

    def node_html_contents(self, node_name: str):
        if not node_name.startswith(self.prefix):
            return f"<p>{self.name()}:{node_name}</p>"
        else:
            return super().node_html_contents(node_name)

