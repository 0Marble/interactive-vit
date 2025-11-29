
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

def compute(request: http.HttpRequest):
    try:
        assert request.method == "POST"

        msg = Message()
        msg.decode(request.body)

        t = msg.get("o")
        h, w = t.shape
        t = t.reshape((1, h, w))

        with torch.no_grad():
            weight = torch.tensor([[[[-1,-1,-1],[2,2,2],[-1,-1,-1]]]], dtype=torch.float)
            conv = torch.nn.Conv2d(1, 1, (3, 3))
            conv.weight = torch.nn.Parameter(weight)
            t = conv(t)
        _, h, w = t.shape
        t = t.reshape((h, w))

        msg.clear()
        msg.set("o", t)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))
