from django.shortcuts import render

# Create your views here.

def glasses(request):
    return render(request,'face/glasses.html')