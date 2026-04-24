from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os


app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

def decode_image(data):
    try:
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decode: {e}")
        return None

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    img = decode_image(data['image'])
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400
        
    feature = data['feature']
    val = int(data.get('value', 128))

    # Fitur 6 citra yang dipilih
    if feature == 'grayscale':
        # Fitur 1: Grayscale
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif feature == 'filter-median':
        # Fitur 2: Filter Median 
        k = 3 if val <= 128 else 5
        result = cv2.medianBlur(img, k)
    elif feature == 'brightness':
        # Fitur 3: Brightness
        offset = val - 128
        result = cv2.convertScaleAbs(img, alpha=1, beta=offset)
    elif feature == 'contrast':
        # Fitur 4: Kontras
        alpha = val / 128.0
        result = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    elif feature == 'binary':
        # Fitur 5: Biner
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    elif feature == 'canny':
        # Fitur 6: Deteksi Tepi (Canny)
        result = cv2.Canny(img, 100, 200)
    else:
        result = img

    return jsonify({'result': encode_image(result)})

if __name__ == '__main__':
    app.run(debug=True)