from __future__ import annotations
from typing import Dict
from urllib.parse import urlencode
import torch

class Node:
    def __init__(self, name: str, params: Dict[str, str], index: int):
        self.name = name
        self.params = params
        self.index = index 

        self.inputs: Dict[str, Edge] = {}
        self.outputs: Dict[str, Edge] = {}

    def get_pinin(self) -> Pinout:
        res = Pinout()
        for ch, e in self.inputs.items():
            assert e.tensor is not None
            res.set(ch, e.tensor)
        return res

    def set_pinout(self, pinout: Pinout):
        for ch, t in pinout.pinout.items():
            if ch in self.outputs:
                self.outputs[ch].tensor = t
            else:
                edge = Edge(Port(self, ch, "out"), None)
                edge.tensor = t
                self.outputs[ch] = edge

    def get_pinout(self) -> Pinout:
        res = Pinout()
        for ch, e in self.outputs.items():
            assert e.tensor is not None
            res.set(ch, e.tensor)
        return res


class Port:
    def __init__(self, node: Node, channel: str, direction: str) -> None:
        self.node = node
        self.channel = channel
        self.direction = direction


class Edge:
    def __init__(self, src: Port | None, tgt: Port | None) -> None:
        if src is not None: assert src.direction == "out"
        if tgt is not None: assert tgt.direction == "in"

        self.input = src 
        self.output = tgt
        self.tensor: None | torch.Tensor = None

class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def add_node(self, name: str, params: Dict[str, str]):
        node = Node(name, params, len(self.nodes))
        self.nodes.append(node)
        return node

    def connect(self, a: Node, a_ch: str, b: Node, b_ch: str):
        a_port = Port(a, a_ch, "out")
        b_port = Port(b, b_ch, "in")
        edge = Edge(a_port, b_port)
        a.outputs[a_ch] = edge
        b.inputs[b_ch] = edge
        return edge

    def add_input(self, value: torch.Tensor, node: Node, channel: str):
        port = Port(node, channel, "in")
        edge = Edge(None, port)
        edge.tensor = value
        node.inputs[channel] = edge
        return edge

    def order(self) -> list[Node]:
        res = []
        visited = set()
        unvisited = [n for n in self.nodes]

        while len(unvisited) != 0:
            x = unvisited.pop()
            all_inputs = True

            for _, edge in x.inputs.items():
                if edge.input is not None and edge.input.node not in visited:
                    all_inputs = False
                    break

            if not all_inputs: 
                unvisited.insert(0, x)
            else:
                visited.add(x)
                res.append(x)

        return res

    def __str__(self) -> str:
        res = "graph:"

        for node in self.nodes:
            name = node.name + "?" + urlencode(node.params)
            for ch, e in node.outputs.items():
                res += "\n\t" + name + " --[" + ch + "]--> "
                if e.output is not None: 
                    res += e.output.node.name + "?" + urlencode(e.output.node.params)
                else:
                    res += "*"

                if e.tensor is not None:
                    res += f" {e.tensor.shape}"

            for ch, e in node.inputs.items():
                if e.input is not None: continue
                assert e.tensor is not None
                res += "\n\t* --[" + ch + "]--> " + name + f" {e.tensor.shape}"

        return res

class Pinout:
    def __init__(self) -> None:
        self.pinout: Dict[str, torch.Tensor] = {}

    def set(self, ch: str, t: torch.Tensor):
        self.pinout[ch] = t

    def get(self, ch: str) -> torch.Tensor | None:
        if ch in self.pinout: return self.pinout[ch]
        return None
