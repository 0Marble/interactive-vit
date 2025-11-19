from django.shortcuts import render
import django.http as http
from django.template import loader

import torch
from PIL import Image
import io
import array
import sys
import logging

logger = logging.getLogger(__name__)

def hello_world(request):
    return http.HttpResponse("Hello, world.")

def index(request):
    template = loader.get_template("main/index.html")
    return http.HttpResponse(template.render({}, request))

def dummy_contents(request):
    return http.HttpResponse("Dummy Node")

def dummy_compute(request: http.HttpRequest):
    try:
        msg = Message()
        msg.decode(request.body)

        ch, dims, data = msg.tensors[0]
        t = torch.tensor(data).reshape(dims.tolist())
        t = torch.cos(t)
        data.clear()
        data.frombytes(t.numpy().tobytes())

        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))


def dummy_description(request):
    return http.JsonResponse([{"kind":"in", "channel":"o", "access":"1"},{"kind":"out", "channel":"o", "access":"*"} ], safe=False)


class Message:
    def __init__(self):
        self.tensors = []

    def encode(self):
        network = "big"
        writer = io.BytesIO()
        writer.write(int.to_bytes(len(self.tensors), 4, network))

        for (channel, dims, data) in self.tensors:
            enc = channel.encode(encoding="utf8")
            writer.write(int.to_bytes(len(enc), 4, network))
            writer.write(enc)

            padding = len(enc) % 4
            if padding != 0: padding = 4 - padding
            writer.write(bytes(padding))

            writer.write(int.to_bytes(len(dims), 4, network))
            writer.write(dims.tobytes())
            writer.write(data.tobytes())

        return writer.getbuffer()

    def decode(self, b: bytes):
        network = "big"
        self.tensors = []

        reader = io.BytesIO(b)
        num_packets = int.from_bytes(reader.read(4), byteorder=network, signed=False)

        for i in range(0, num_packets):
            channel_len = int.from_bytes(reader.read(4), byteorder=network, signed=False)
            channel = reader.read(channel_len).decode(encoding="utf8")
            
            padding = channel_len % 4
            if padding != 0: padding = 4 - padding
            reader.read(padding)

            n_dim = int.from_bytes(reader.read(4), byteorder=network, signed=False)
            dims = array.array('I')
            dims.frombytes(reader.read(4 * n_dim))

            elem_cnt = 1
            for d in dims: elem_cnt *= d

            data = array.array('f')
            data.frombytes(reader.read(4 * elem_cnt))

            self.tensors.append((channel, dims, data))

