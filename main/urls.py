from django.urls import path

from . import views

urlpatterns = [
    path("hello", views.hello_world, name="hello_world"),
    path("", views.index, name="index"),
    path("dummy/description", views.dummy_description, name="dummy-description"),
    path("dummy/contents", views.dummy_contents, name="dummy-contents"),
] 
