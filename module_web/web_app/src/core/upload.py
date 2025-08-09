from flask import request, jsonify, send_file, make_response
import cv2
import numpy as np
import io
import os

from BladeDamageConfiguration import BladeDamageConfiguration
from module_predict.ImagePredict import ImagePredict


def upload_file():
    print("Solicitação de upload de arquivo.")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']


    filename = file.filename
    name, ext = os.path.splitext(filename)
    download_name = f"{name}_predict{ext}"

    # Leitura da imagem original
    in_memory_file = io.BytesIO(file.read())
    img = cv2.imdecode(np.frombuffer(in_memory_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processamento da imagem
    processed_img, text_prediction = process_image(img, request.form)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

    _, img_encoded = cv2.imencode('.png', processed_img)
    img_bytes = img_encoded.tobytes()

    print(f"Imagem processada com sucesso ({filename}).")
    response = make_response(send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=False,
            download_name=download_name
        ))
    response.headers['X-Download-Filename'] = download_name
    response.headers['text-prediction'] = text_prediction
    return response


def process_image(image, form):

    print("Solicitação de processamento de imagem.")

    # Define configuração
    bdconf = BladeDamageConfiguration('/Users/albertoregio/Downloads/')

    # Cria preditor
    ip = ImagePredict(image, bdconf)

    # Define propriedades da imagem
    ip.predict_flag = form.get('show_boundingbox') == 'on'
    # ip.prediction = None
    ip.text_predict_flag = form.get('show_text_predict') == 'on'
    ip.text_percentual = form.get('show_percentual') == 'on'
    ip.text_layout = form.get('show_all_results') == 'on'
    ip.heatmap_flag = form.get('show_heatmap') == 'on'
    # ip.alpha = 1.0

    # Gera uma imagem
    img_prediction = ip.get_image()
    text_prediction = ip.prediction[0] if ip.prediction is not None else "No prediction"
    return img_prediction, text_prediction
