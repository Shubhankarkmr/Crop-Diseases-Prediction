from django.db import models

# Create your models here.
class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True)
    treatment = models.TextField()

    def __str__(self):
        return self.name