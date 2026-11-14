from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image, ImageOps  
import numpy as np

app = Flask(__name__)

print("Cargando el modelo de IA")
modelo = tf.keras.models.load_model('modelo.h5')
print("Modelo cargado")

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predecir', methods=['POST'])  
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'ERROR'})

    archivo = request.files['imagen']

    if archivo.filename == '':
        return jsonify({'error' : 'ARCHIVO NO CARGADO'})

    try:
        img = Image.open(archivo.stream).convert('L')
        
        img_invertida = ImageOps.invert(img)
        img_redimensionada = img_invertida.resize((28, 28))

        img_array = np.array(img_redimensionada) / 255.0
        img_lista = np.expand_dims(img_array, axis=0)

        predicciones = modelo.predict(img_lista)
        numero_predicho = int(np.argmax(predicciones[0]))
        
        confianza = float(predicciones[0][numero_predicho] * 100)

        return jsonify({
            'numero_predicho': numero_predicho,
            'confianza': round(confianza, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)