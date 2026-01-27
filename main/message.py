import array
import io
import json
import os
import torch
import logging

from main.graph import Graph

logger = logging.getLogger(__name__)


def align_next(offset, align):
    m = offset % align
    if m == 0: return offset
    return offset + align - m

class Request:
    def __init__(self):
        self.graph = Graph()

    def decode(self, b: bytes):
        reader = io.BytesIO(b)
        byte_size = int.from_bytes(reader.read(4), "little")
        assert int.from_bytes(reader.read(4), "little") == 0x69babe69
        block_cnt = int.from_bytes(reader.read(4), "little")
        json_size = int.from_bytes(reader.read(4), "little")

        json_str = reader.read(json_size).decode(encoding="utf-8")
        json_obj = json.loads(json_str)

        offset = reader.tell()
        padding = align_next(offset, 4) - offset
        _ = reader.read(padding) 
        assert reader.tell() % 4 == 0

        logger.info("decode message: size=%d, json_size=%d, padding=%d, block_cnt=%d", byte_size, json_size, padding, block_cnt)
        logger.info("json: %s", json_str)

        tensors: list[torch.Tensor] = []
        for i in range(0, block_cnt):
            start = reader.tell()

            block_size = int.from_bytes(reader.read(4), "little")
            dim_cnt = int.from_bytes(reader.read(4), "little")

            dims = array.array("I")
            dims.frombytes(reader.read(4 * dim_cnt))

            elem_cnt = 1
            for x in dims: elem_cnt *= x

            data = array.array("f")
            data.frombytes(reader.read(4 * elem_cnt))
            logger.info("tensor %d: size=%d, dim_cnt=%d dims=%s", i, block_size, dim_cnt, f"{dims.tolist()}")

            assert start + block_size == reader.tell()
            t = torch.tensor(data).reshape(dims.tolist())
            tensors.append(t)

        for node_json in json_obj["nodes"]:
            _ = self.graph.add_node(node_json["endpoint"], node_json["params"])

        for edge_json in json_obj["edges"]:
            tgt_node = self.graph.nodes[edge_json["out_port"]["node"]]
            tgt_ch = edge_json["out_port"]["channel"]

            if "tensor" in edge_json:
                _ = self.graph.add_input(tensors[edge_json["tensor"]], tgt_node, tgt_ch)
            else:
                src_node = self.graph.nodes[edge_json["in_port"]["node"]]
                src_ch = edge_json["in_port"]["channel"]
                _ = self.graph.connect(src_node, src_ch, tgt_node, tgt_ch)


class Response:
    def __init__(self):
        self.outputs = {}

    def set_output(self, node: int, channel: str, t: torch.Tensor):
        if node not in self.outputs: self.outputs[node] = {}
        self.outputs[node][channel] = t

    def encode(self) -> bytes:
        writer = io.BytesIO()

        json_obj = []
        tensors: list[torch.Tensor] = []
        for node in self.outputs.keys():
            outputs = self.outputs[node]
            for channel in outputs.keys():
                json_obj.append({"node": node, "channel": channel})
                tensors.append(outputs[channel])
        
        json_utf8 = json.dumps(json_obj).encode()

        writer.write(int.to_bytes(0, 4, "little")) # byte_size to patch
        writer.write(int.to_bytes(0xdeadbeef, 4, "little")) # magic
        writer.write(int.to_bytes(len(tensors), 4, "little")) # block_cnt 
        writer.write(int.to_bytes(len(json_utf8), 4, "little")) # json size

        writer.write(json_utf8)
        offset = writer.tell()
        writer.seek(align_next(offset, 4) - offset, os.SEEK_CUR)

        for t in tensors:
            dims = array.array("I")
            dims.fromlist(list(t.shape))
            data = array.array("f")
            data.frombytes(t.numpy().tobytes())
            block_size = 4 + 4 + len(dims) * 4 + len(data) * 4

            writer.write(int.to_bytes(block_size, 4, "little"))
            writer.write(int.to_bytes(len(dims), 4, "little"))
            writer.write(dims.tobytes())
            writer.write(data.tobytes())

        byte_size = writer.tell()
        writer.seek(0)
        writer.write(int.to_bytes(byte_size, 4, "little"))

        return writer.getbuffer().tobytes()
