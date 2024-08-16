from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mindspring.h5')

# Define class names (adjust according to your dataset)
class_names = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions',
    'Eczema', 'Melanocytic Nevi (NV)', 'Melanoma',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections'
]

# Define the upload folderins
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_skin_disease(img_path):
    processed_img = load_and_preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_disease = None
    confidence = None
    uploaded_image_url = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_disease, confidence = predict_skin_disease(file_path)
            uploaded_image_url = file.filename  # Just the filename, not the full path

    return render_template('index.html',
                           predicted_disease=predicted_disease,
                           confidence=confidence,
                           uploaded_image_url=uploaded_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)