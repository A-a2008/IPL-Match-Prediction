from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("how-to-use", views.how_to_use, name="how_to_use")
]
