# Import Library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Memuat model
model = keras.models.load_model("model.h5")

# Label model
label = ['Bekantan', 'Kakatua Jambul Kuning', 'Komodo', 'Whale Shark']

app = Flask(__name__)

# Fungsi untuk memprediksi label
def predict_label(img):
    i = np.asarray(img) / 255.0
    # Ukuran input model
    i = i.reshape(1, 224, 224, 3)  # Pastikan ini sesuai dengan ukuran input yang diharapkan model
    pred = model.predict(i)  # Diperbaiki dari predic menjadi predict
    result = label[np.argmax(pred)]
    return result

@app.route("/predict", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((244, 244), Image.NEAREST)  # Ubah ukuran untuk sesuai dengan input model
    pred_img = predict_label(img)
    return jsonify({"prediction": pred_img})  # Kembalikan prediksi sebagai JSON

if __name__ == "__main__":
    app.run(debug=True)
    