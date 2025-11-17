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

def input(request: http.request.HttpRequest):
    img_data = request.FILES["image_input"]
    img = Image.open(img_data).convert("RGB")
    out = img.transpose(Image.Transpose.ROTATE_90)
    
    out_writer = io.BytesIO()
    out.save(out_writer, format="jpeg")
    out_writer.seek(0)

    resp = http.HttpResponse(out_writer.getvalue(), content_type = img_data.content_type)

    return resp



