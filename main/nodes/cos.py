from django.shortcuts import render
import django.http as http
from django.template import loader
import logging
import torch

from main.message import Request

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
