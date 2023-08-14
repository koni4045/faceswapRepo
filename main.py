import os

import cv2
import insightface
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from insightface.app import FaceAnalysis

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/faces/<filename>')
def uploaded_faces(filename):
    return send_from_directory('faces', filename)


@app.route('/face_detect_and_select', methods=['GET', 'POST'])
def display_faces():
    clear_folder('faces')
    clear_folder('uploads')
    if 'source' not in request.files:
        return "No image part in the form"

    source = request.files['source']
    if source.filename == '':
        return "No selected image"

    source.save('uploads/' + source.filename)

    faces = detect_faces('uploads/' + source.filename)
    filenames = []
    for i, face in enumerate(faces):
        cropped_image = (Image.open('uploads/' + source.filename)).crop(face['bbox'])
        filename = 'faces/face' + str(i + 1) + '.jpg'
        filenames.append('face' + str(i + 1) + '.jpg')
        cropped_image.save(filename)
    print(filenames)
    # result_image = swap_faces(detect_faces('uploads/' + target.filename), source_face, 'uploads/' + target.filename)
    # cv2.imwrite('uploads/result.jpg', result_image)
    return render_template('face_detect_and_select.html', filenames=filenames)


@app.route('/target_face_upload_and_swap', methods=['GET', 'POST'])
def target_face_upload_and_swap():
    face_number = -1
    if request.method == 'POST':
        selected_value = request.form.get('selected_file')  # Get the selected dropdown value
        face_number = int(selected_value[-1]) - 1
        print(face_number)
        # Use the selected_value as needed
    if 'target' not in request.files:
        return "No image part in the form"
    print(os.listdir('uploads')[0])
#source = Image.open('uploads/' + os.listdir('uploads')[0])
    source_face = detect_faces('uploads/' + os.listdir('uploads')[0])[face_number]
    target = request.files['target']
    if target.filename == '':
        return "No selected image"

    target.save('uploads/' + target.filename)

    result_image = swap_faces(detect_faces('uploads/' + target.filename), source_face, 'uploads/' + target.filename)
    cv2.imwrite('uploads/result.jpg', result_image)
    return render_template('result_image.html', target_path=target.filename,
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


def clear_folder(folder_path):
    try:
        # List all files in the folder
        file_list = os.listdir(folder_path)

        # Loop through the files and delete each one
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return True  # Successfully cleared all images
    except Exception as e:
        return False  # An error occurred


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
