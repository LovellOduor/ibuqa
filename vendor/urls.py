from . import views
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('become_a_vendor/',views.become_a_vendor,name='become_a_vendor'),
    path('vendor_admin/',views.vendor_admin,name='vendor_admin'),
    path('add_product/',views.add_product,name='add_product'),
    path('update_profile/',views.update_info,name='update_profile'),
    path('logout/', auth_views.LogoutView.as_view(),name='logout'),
    path('login/', auth_views.LoginView.as_view(template_name='vendor/login.html'),name="login")
]