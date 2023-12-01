from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from .image_processing import process_image
import os
import base64
from PIL import Image
from io import BytesIO
import cv2
from flask import send_file

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    @app.route('/', methods=['GET'])
    def render_main():
        return render_template('main.html')

    @app.route('/process_image', methods=['POST'])
    def process_image_route():
        try:
            input_image_base64 = request.form['inputImagePath']
            selected_algorithm = request.form['algorithm']

            # Chuyển đổi base64 thành hình ảnh và lưu vào thư mục uploads
            img_data = base64.b64decode(input_image_base64.split(',')[1])
            img = Image.open(BytesIO(img_data))
            filename = secure_filename('input_image.jpg')
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(input_image_path)

            processed_image = process_image(input_image_path, selected_algorithm)

            result_filename = 'processed_image.jpg'
            # result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], result_filename))
            cv2.imwrite(result_path, processed_image)

            # Trả về file hình ảnh đã xử lý bằng send_file
            return send_file(result_path, mimetype='image/jpeg', as_attachment=False)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
