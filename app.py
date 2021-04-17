#import pytesseract
import os
from flask import Flask, render_template, request

# import OCR function
from ocr import ocr_core

# Define a folder to store and later serve the images
UPLOAD_FOLDER = '/static/uploads/'

# Allow files of a specific type
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)

# Function to check the file extension
def allowed_file(filename):
    x = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return x

@app.route('/')
def home_page():
    return render_template('index.html')

# route and function to handle the upload image
@app.route('/upload', methods = ['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg = 'No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg = 'No file selected')
        
        if file and allowed_file(file.filename):

            # call the OCR function on it
            extracted_text = ocr_core(file)

            # extract the text and display it
            return render_template('upload.html', msg = 'Success', extracted_text = extracted_text, img_src = UPLOAD_FOLDER + file.filename)

    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    app.run()

