import os

import cv2
import insightface
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from insightface.app import FaceAnalysis

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/imageDisplay', methods=['GET', 'POST'])
def display_image():
    if 'source' not in request.files:
        return "No image part in the form"

    source = request.files['source']
    if source.filename == '':
        return "No selected image"

    # You can process the image here or save it to a desired location
    # For example, to save the image to the 'uploads' folder:
    source.save('uploads/' + source.filename)

    if 'target' not in request.files:
        return "No image part in the form"

    target = request.files['target']
    if target.filename == '':
        return "No selected image"

    # You can process the image here or save it to a desired location
    # For example, to save the image to the 'uploads' folder:
    target.save('uploads/' + target.filename)
    faces = detect_faces('uploads/' + source.filename)
    source_face = faces[0]
    result_image = swap_faces(detect_faces('uploads/' + target.filename), source_face, 'uploads/' + target.filename)
    cv2.imwrite('uploads/result.jpg', result_image)
    return render_template('image_display.html', source_path=source.filename, target_path=target.filename,
                           image_filename='result.jpg')


def detect_faces(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    fapp = FaceAnalysis(name='buffalo_l')
    fapp.prepare(ctx_id=0, det_size=(640, 640))
    faces = fapp.get(image)
    return faces


def swap_faces(faces, source_face, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=False)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    res = image.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
