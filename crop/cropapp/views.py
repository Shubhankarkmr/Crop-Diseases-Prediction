from django.shortcuts import render
import torch
from .forms import ImageUploadForm
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import os
import warnings
from django.conf import settings
from .models import Disease
from googletrans import Translator

def home(request):
    return render(request,"cropapp\homepage.html")

# Suppress warnings
warnings.filterwarnings("ignore")

# Path to your saved model (.pt file)
MODEL_PATH = r"E:\Downloads\densenet201____plantvillage___scratch.pt"

# Define the class names (adjust according to your training classes)
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', 'Cherry(including_sour)healthy', 'Cherry(including_sour)Powdery_mildew',
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)healthy',
    'Corn(maize)Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__healthy',
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot',
    'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight',
    'Potato_healthy', 'Potato_Late_blight', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
    'Strawberry_healthy', 'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus'
]

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        # Attempt to load the full model directly
        model = torch.load(MODEL_PATH, map_location=device)
        model = model.to(device)
        model.eval()
        
    except TypeError:
        # Load only the state dictionary
        
        model = models.densenet201(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(class_names))
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

def translate_text(text, dest_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        return "Translation Error: " + str(e)

# View to handle the image upload
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

            with open(image_path, 'wb') as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)

            model = load_model()
            predicted_disease = predict_image(image_path, model)

            # Fetch treatment information from the database
            try:
                disease_obj = Disease.objects.get(name=predicted_disease)
                treatment_info = disease_obj.treatment
            except Disease.DoesNotExist:
                treatment_info = "Treatment information not available."
            
            dest_language = request.POST.get('language', 'en')  # Default to English
            translated_treatment = translate_text(treatment_info, dest_language)
            translated_label = translate_text("Translated Preventive Measures :", dest_language)

            return render(request, 'cropapp/result.html', {
                'disease': predicted_disease,
                'image_url': os.path.join(settings.MEDIA_URL, uploaded_image.name),
                'treatment_info': treatment_info,'translated_treatment': translated_treatment,'translated_label': translated_label,
                'language': dest_language
            })
            
    else:
        form = ImageUploadForm()
    return render(request, 'cropapp/upload.html', {'form': form})