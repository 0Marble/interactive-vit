from typing import Dict, Tuple

import torch
from main.context import NodeKind
from main.graph import Pinout

class CosNode(NodeKind):
    def __init__(self):
        super().__init__("cos")

    def decode_params(self, params: Dict[str, str]) -> Tuple[float, float]:
        a = 1.0
        if "A" in params: a = float(params["A"])
        b = 0.0
        if "b" in params: b = float(params["b"])
        return a, b

    def contents(self, params: Dict[str, str]) -> str:
        a, b = self.decode_params(params)
        return f"cos({a}x+{b})"

    def io(self, params: Dict[str, str]) -> Dict:
        _ = params
        return {"ins": ["o"], "outs": ["o"]}

    def compute(self, params: Dict[str, str], inputs: Pinout) -> Pinout:
        a, b = self.decode_params(params)
        x = inputs.get("o")
        if x is None: raise Exception("missing input: o")
        y = torch.cos(a * x + b)

        res = Pinout()
        res.set("o", y)
        return res

def instances():
    return [CosNode()]
