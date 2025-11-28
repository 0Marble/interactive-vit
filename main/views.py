from django.shortcuts import render
import django.http as http
from django.template import loader

import torch
from PIL import Image
import logging

from . import message 

import random

logger = logging.getLogger(__name__)

def index(request):
    template = loader.get_template("main/index.html")
    return http.HttpResponse(template.render({}, request))

