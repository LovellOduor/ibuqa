from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
urlpatterns = [
    path('', views.homepage,name='home-page'),
    path('contact/',views.contactpage,name='contact-page')
]