from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os

model = VGG16(weights='imagenet')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))

            img_array = preprocess_input(img_array)

            result = model.predict(img_array)
            prediction = decode_predictions(result, top=5)[0]

            formatted_prediction = [
                {'label': pred[1], 'confidence': pred[2]} for pred in prediction
            ]
            
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': formatted_prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
