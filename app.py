import os
from handwriting_processing import evaluate, OUTPUT_FOLDER
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

#Initialize the Flask app
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def evaluate_image():
    if request.method == 'POST':
        if request.form.get('uploadImg') == 'Upload Image':
            input_image = request.files['image']

            if input_image != '':   #user uploaded an image
                filename = secure_filename(input_image.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                input_image.save(file_path)
                text_output = evaluate(file_path)

                #clean up uploaded image - it's not needed anymore
                if os.path.exists(file_path):
                    os.remove(file_path)

                return render_template('prediction.html', converted_text=text_output)
        elif request.form.get('downloadDefault') == 'Default Image':
            return send_from_directory(app.root_path, "default_test.jpg")
    return render_template('index.html')


@app.route('/processed_img', methods=['GET', 'POST'])
def send_processed_img():
    return send_from_directory(app.config["OUTPUT_FOLDER"], "thresholded.png")

if __name__ == "__main__":
    app.run()
    





