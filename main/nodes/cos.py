from django.shortcuts import render
import django.http as http
from django.template import loader
import torch

from main.message import Message

def description(request):
    return http.JsonResponse([
        {"kind":"in", "channel":"o", "access":"1"},
        {"kind":"out", "channel":"o", "access":"*"},
    ], safe=False)

def contents(request: http.HttpRequest):
    params = request.GET
    A = params.get("A") 
    if A is None: A = 1.0
    else: A = float(A)

    b = params.get("b")
    if b is None: b = 0.0
    else: b = float(b)

    return http.HttpResponse(f"cos({A}x+{b})")

# computes cos(Ax+b)
def compute(request: http.HttpRequest):
    try:
        assert request.method == "POST"

        msg = Message()
        msg.decode(request.body)
        params = request.POST

        A = params.get("A") 
        if A is None: A = 1.0
        else: A = float(A)

        b = params.get("b")
        if b is None: b = 0.0
        else: b = float(b)

        dims, data = msg.get("o")

        t = torch.tensor(data).reshape(dims.tolist())
        t = torch.cos(A * t + b)
        data.clear()
        data.frombytes(t.numpy().tobytes())

        msg.clear()
        msg.set("o", dims, data)
        res = msg.encode()
        return http.HttpResponse(res)
    except Exception as e:
        return http.HttpResponseBadRequest(str(e))
