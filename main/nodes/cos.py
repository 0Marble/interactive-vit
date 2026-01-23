from django.shortcuts import render
import django.http as http
from django.template import loader
import logging
import torch

from main.message import Message

logger = logging.getLogger(__name__)

def description(request):
    return http.JsonResponse({"ins": ["o"], "outs": ["o"]}, safe=False)

def contents(request: http.HttpRequest):
    params = request.GET
    A = params.get("A") 
    if A is None: A = 1.0
    else: A = float(A)

    b = params.get("b")
    if b is None: b = 0.0
    else: b = float(b)

    return http.HttpResponse(f"cos({A}x+{b})")

def compute(request: http.HttpRequest):
    try:
        assert request.method == "POST"

        msg = Message()
        msg.decode(request.body)
        params = request.GET

        A = params.get("A") 
        if A is None: A = 1.0
        else: A = float(A)

        b = params.get("b")
        if b is None: b = 0.0
        else: b = float(b)

        t = msg.get("o")

        t = torch.cos(A * t + b)

        msg.clear()
        msg.set("o", t)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))
