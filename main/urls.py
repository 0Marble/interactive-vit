from django.urls import path as django_path
from django.conf import settings

from . import views

import importlib
import os
import logging
import sys

import main.nodes.cos as cos

logger = logging.getLogger(__name__)

urlpatterns = [
    django_path("", views.index, name="index"),
    django_path("list_graphs", views.list_graphs, name="list-graphs"),
    django_path("load_graph", views.load_graph, name="load-graph"),
    django_path("compute", views.compute, name="compute"),
    django_path("node/cos/description", cos.description, name="cos-description"),
    django_path("node/cos/contents", cos.contents, name="cos-contents"),
]
