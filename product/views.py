from django.shortcuts import render, get_object_or_404, redirect
from .models import Category, Product
from django.core.paginator import Paginator
from django.db.models import Q
from django.contrib import messages
from cart.cart import Cart
from .forms import AddToCartForm
# Create your views here.

def product(request,category_slug,product_slug):
    cart = Cart(request)
    product = get_object_or_404(Product,category__slug=category_slug,slug=product_slug)
    if request.method == "POST":
        form = AddToCartForm(request.POST)
        if form.is_valid():
            quantity = form.cleaned_data['quantity']
            cart.add(product_id=product.id,quantity=quantity,update_quantity=False)
            messages.success(request, 'Added To Cart')
            return redirect('product',category_slug=category_slug,product_slug=product_slug)
    else:
        form = AddToCartForm()

    return render(request,'product/product.html',{'product':product,'form':form})
    
def search(request):
    query = request.GET.get('query','')
    products = Product.objects.filter(Q(title__icontains=query)| Q(description__icontains=query))
    products_paginator = Paginator(products,1)
    page_num = request.GET.get('page')
    page = products_paginator.get_page(page_num)
    return render(request,'core/home.html',{'page':page,'query':query})