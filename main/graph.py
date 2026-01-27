from typing import Dict
import torch

class Node:
    def __init__(self, name: str, params, index: int):
        self.name = name
        self.params = params
        self.index = index 

        self.inputs: Dict[str, Edge] = {}
        self.outputs: Dict[str, Edge] = {}

class Port:
    def __init__(self, node: Node, channel: str, direction: str) -> None:
        self.node = node
        self.channel = channel
        self.direction = direction


class Edge:
    def __init__(self, src: Port | None, tgt: Port) -> None:
        if src is not None: assert src.direction == "out"
        assert tgt.direction == "in"

        self.input = src 
        self.output = tgt
        self.tensor: None | torch.Tensor = None

class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def add_node(self, name: str, params):
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
        unvisited = self.nodes

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

