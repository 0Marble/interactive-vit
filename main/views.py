from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def hello_world(request):
    return HttpResponse("Hello, world.")

def index(request):
    template = loader.get_template("main/index.html")
    return HttpResponse(template.render({}, request))
