from django.shortcuts import render
import django.http as http
from django.template import loader

import torch
from PIL import Image
import io

def hello_world(request):
    return http.HttpResponse("Hello, world.")

def index(request):
    template = loader.get_template("main/index.html")
    return http.HttpResponse(template.render({}, request))

def dummy_contents(request):
    return http.HttpResponse("Dummy Node")

def dummy_description(request):
    return http.JsonResponse([{"kind":"in", "channel":"o", "access":"1"},{"kind":"out", "channel":"o", "access":"*"} ], safe=False)


