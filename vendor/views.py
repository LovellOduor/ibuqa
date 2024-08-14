from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.utils.text import slugify
from .models import Vendor
from product.models import Product
from .forms import ProductForm, VendorForm

def become_a_vendor(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request,user)
            vendor = Vendor.objects.create(name=user.username,created_by=user)
            return redirect('home-page')
    else:
        form = UserCreationForm()
    return render(request,'vendor/become_a_vendor.html',{'form':form})

@login_required
def vendor_admin(request):
    vendor = request.user.vendor
    products = vendor.products.all()
    return render(request,'vendor/vendor_admin.html',{'vendor':vendor,'products':products})

@login_required
def add_product(request):
    if request.method == "POST":
        form = ProductForm(request.POST,request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.vendor = request.user.vendor
            product.slug = slugify(product.title)
            product.save()
            return redirect('vendor_admin')
    else:
        form = ProductForm()
    return render(request,'vendor/add_product.html',{'form':form})

@login_required
def update_info(request):
    instance = get_object_or_404(Vendor,name=request.user.vendor.name)
    user = get_object_or_404(User,username=request.user.vendor.name)
    form = VendorForm(request.POST or None, instance=instance)
    if request.method == "POST":
        if form.is_valid():
            updateduser = form.save()
            user.username = updateduser.name
            user.save()
            return redirect('home-page')
        else:
            form = VendorForm()
    return render(request,'vendor/update_profile.html',{'form':form})
