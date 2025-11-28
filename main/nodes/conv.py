
from django.shortcuts import render
import django.http as http
from django.template import loader
import torch

from main.message import Message

import array

def description(request):
    return http.JsonResponse([
        {"kind":"in", "channel":"o", "access":"1"},
        {"kind":"out", "channel":"o", "access":"*"},
    ], safe=False)

def contents(request: http.HttpRequest):
    return http.HttpResponse("Convolution")

# computes cos(Ax+b)
def compute(request: http.HttpRequest):
    try:
        assert request.method == "POST"

        msg = Message()
        msg.decode(request.body)

        dims, data = msg.get("o")

        dims_list = dims.tolist()
        dims_list.insert(0, 1) # [C=1, H, W]
        t = torch.tensor(data).reshape(dims_list)

        with torch.no_grad():
            weight = torch.tensor([[[[-1,-1,-1],[2,2,2],[-1,-1,-1]]]], dtype=torch.float)
            conv = torch.nn.Conv2d(1, 1, (3, 3))
            conv.weight = torch.nn.Parameter(weight)
            t = conv(t)
        _, h, w = t.shape

        data.clear()
        data.frombytes(t.numpy().tobytes())

        msg.clear()
        out_dims = array.array("I")
        out_dims.fromlist([h, w])
        msg.set("o", out_dims, data)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))
