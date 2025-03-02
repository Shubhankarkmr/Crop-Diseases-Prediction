# forms.py
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select a crop image to predict disease')
