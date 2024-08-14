from django.urls import path
from . import views

urlpatterns = [
    path('',views.glasses ,name='glasses')
]