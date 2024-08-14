from django.urls import path
from .views import handapi ,faceapi

urlpatterns = [
    path("hand",handapi.as_view(),name="hand"),
    path("face",faceapi.as_view(),name="face")
]
