from torchvision.models import vgg16
import torch

from main.message import Message

class Model:
    def __init__(self):
        self.model = vgg16()

    def name(self):
        return "vgg16"

    def list_node_names(self):
        names = [
                'features.0', 
                'features.1', 
                'features.2', 
                'features.3', 
                'features.4', 
                'features.5', 
                'features.6', 
                'features.7', 
                'features.8', 
                'features.9', 
                'features.10', 
                'features.11', 
                'features.12', 
                'features.13', 
                'features.14', 
                'features.15', 
                'features.16', 
                'features.17',
                'features.18',
                'features.19',
                'features.20',
                'features.21',
                'features.22',
                'features.23',
                'features.24',
                'features.25',
                'features.26',
                'features.27',
                'features.28',
                'features.29',
                'features.30',
                'avgpool',
                'classifier.0',
                'classifier.1',
                'classifier.2',
                'classifier.3',
                'classifier.4',
                'classifier.5',
                'classifier.6',
        ]
        return names

    def eval_node(self, node_name: str, in_msg: Message) -> Message:
        with torch.no_grad():
            sub = self.model.get_submodule(node_name)
            res = sub(in_msg.get("o"))
            assert isinstance(res, torch.Tensor)

            out_msg = Message()
            out_msg.set("o", res)
            return out_msg 

    def node_html_contents(self, node_name):
        return f"<p>{node_name}<p>"

    def node_io_description(self, node_name):
        return [
            {"kind":"in", "channel":"o", "access":"1"}, 
            {"kind":"out", "channel":"o", "access":"*"}, 
        ]
