import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
import io

app = Flask(__name__)

# 1. LOAD MODEL
# Pastikan nama file sama persis dengan yang ada di folder
MODEL_PATH = 'model_buah_mobilenet32.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# 2. DAFTAR KELAS (PENTING: Sesuaikan urutan dengan saat training!)
# Contoh urutan alfabetis (Ganti jika urutanmu beda)
CLASSES = [
    'Fresh Apple', 
    'Fresh Banana', 
    'Fresh Orange',
    'Rotten Apple', 
    'Rotten Banana', 
    'Rotten Orange'
]

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalisasi (Sesuai training MobileNetV2 umumnya)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima data gambar base64 dari HTML
        data = request.get_json()
        image_data = data['image']
        
        # Decode gambar
        header, encoded = image_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        
        # Proses ke AI
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)
        
        # Ambil hasil tertinggi
        result_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]) * 100)
        label = CLASSES[result_idx]
        
        return jsonify({
            'status': 'success',
            'label': label,
            'confidence': f"{confidence:.1f}%",
            'is_fresh': 'Fresh' in label
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)