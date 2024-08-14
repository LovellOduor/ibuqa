from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Vendor(models.Model):
    name = models.CharField(max_length=256)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.OneToOneField(User,related_name='vendor',on_delete=models.CASCADE)
    approved = models.BooleanField(default=False)
    contact = models.CharField(max_length=256)
    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name