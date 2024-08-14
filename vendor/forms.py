from django.forms import ModelForm
from product.models import Product, Vendor

class ProductForm(ModelForm):
    class Meta:
        model = Product
        fields = ['category','image','image1','image2','image3','title','description','price']

class VendorForm(ModelForm):
    class Meta:
        model = Vendor
        fields = ['name','contact']