import torch
from main.message import Message
import math

class Model:
    def __init__(self, model: torch.nn.Module, prefix: str):
        self.model = model
        self.model.eval()
        self.prefix = prefix

        self.node_names = []
        for (name, sub) in self.model.named_modules():
            if sum([1 for _ in sub.named_modules()]) != 1: continue
            self.node_names.append(self.prefix + name)

    def name(self):
        raise Exception("Model.name: unimplemented")

    def generate_graph(self):
        json_obj = {"nodes": [], "edges": []};
        nodes = self.list_node_names()
        cnt = len(nodes)
        w = int(math.sqrt(cnt))
        for i, name in enumerate(nodes):
            json_obj["nodes"].append({
                "instance":{"kind":"net_node","endpoint":f"{self.name()}/{name}","params":{}},
                "pos":{"x": (i % w) * 200,"y": int(i / w) * 200}
            })

            if i != 0: 
                json_obj["edges"].append({
                    "in_port": {"node": i - 1, "channel": "o"},
                    "out_port": {"node": i, "channel": "o"}
                });

        return json_obj


    def list_node_names(self):
        return self.node_names

    def eval_node(self, node_name: str, in_msg: Message) -> Message:
        with torch.no_grad():
            sub = self.model.get_submodule(node_name.removeprefix(self.prefix))
            res = sub(in_msg.get("o"))
            assert isinstance(res, torch.Tensor)
            out_msg = Message()
            out_msg.set("o", res)
            return out_msg 

    def node_html_contents(self, node_name: str):
        sub = self.model.get_submodule(node_name.removeprefix(self.prefix))
        return f"<p>{self.name()}:{node_name}</p> <p>{sub._get_name()}</p>"

    def node_io_description(self, node_name):
        return [
            {"kind":"in", "channel":"o", "access":"1"}, 
            {"kind":"out", "channel":"o", "access":"*"}, 
        ]
