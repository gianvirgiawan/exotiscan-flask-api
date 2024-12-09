from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Muat model
model = load_model('model.h5')

# Daftar nama kelas
class_names = ['Bekantan', 'Kakatua Jambul Kuning', 'Komodo', 'Whale Shark']

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).resize((224, 224))
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
    return img

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = file.read()
        img = preprocess_image(image)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        class_name = class_names[predicted_class]  # Dapatkan nama kelas dari daftar

        return jsonify({
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'class_name': class_name  # Tambahkan nama kelas dalam output
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
