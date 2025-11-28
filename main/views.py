from django.shortcuts import render
import django.http as http
from django.template import loader

import torch
from PIL import Image
import logging

from . import message 

import random

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
        msg = message.Message()
        msg.decode(request.body)

        dims, data = msg.get("o")

        t = torch.tensor(data).reshape(dims.tolist())
        t = torch.cos(t)
        data.clear()
        data.frombytes(t.numpy().tobytes())

        msg.clear()
        msg.set("o", dims, data)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))


def dummy_description(request):
    return http.JsonResponse([{"kind":"in", "channel":"o", "access":"1"},{"kind":"out", "channel":"o", "access":"*"} ], safe=False)

