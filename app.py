from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Cargar modelo (usamos el nano para que sea rápido en servidor)
model = YOLO('yolo11n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # 1. Obtener la imagen enviada por JavaScript
    file = request.files['image']
    
    # 2. Convertir el archivo a un formato que OpenCV entienda
    # Leemos los bytes y los convertimos a un array de numpy
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 3. Procesar con YOLO
    # conf=0.5: solo detecciones seguras
    # classes=[0]: solo personas
    results = model(img, classes=[0], conf=0.5, verbose=False)

    # 4. Dibujar las cajas en la imagen
    annotated_frame = results[0].plot()

    # 5. Convertir la imagen procesada de nuevo a formato enviada (Base64)
    # Primero codificamos a JPG
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    # Luego convertimos a string base64 para enviarlo fácil por JSON
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_base64})

if __name__ == '__main__':
    # Importante: threaded=True para manejar varias peticiones a la vez
    app.run(debug=True, threaded=True)