from django.shortcuts import render
from product.models import Product
from django.core.paginator import Paginator

# Create your views here.

def homepage(request):
    products = Product.objects.all()[0:8]
    products_paginator = Paginator(products,2)
    page_num = request.GET.get('page')
    page = products_paginator.get_page(page_num)

    return render(request,'core/home.html',{'page':page})

def contactpage(request):
    return render(request,'core/contact.html')