from django.urls import path as django_path
from . import views
import logging

logger = logging.getLogger(__name__)

urlpatterns = [
    django_path("", views.index, name="index"),
    django_path("list_graphs", views.list_graphs, name="list_graphs"),
    django_path("load_graph/<str:name>", views.load_graph, name="load_graph"),
    django_path("compute", views.compute, name="compute"),
    django_path("description/<str:name>", views.description, name="description"),
    django_path("contents/<str:name>", views.contents, name="contents"),
]
