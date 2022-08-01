from asyncore import file_dispatcher
import os
import random
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.debug = True


@app.route('/', methods=['GET'])
def dropdown():
    return render_template('index.html')

@app.route('/index')
def upload_form():
	return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    path = 'static/preInference'
    global random_filename 
    random_filename = random.choice([
        x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x))
    ])
    global file_path
    file_path = path + '/' + random_filename
    return render_template('loading.html', file_path = file_path)

@app.route('/load', methods = ['GET', 'POST'])
def load_page():
    path = 'static/postInference'
    new_file_path = path + '/' + random_filename
    return render_template('inference.html', new_file_path = new_file_path)

#@app.route('/<filename>')
#def display_image(filename):
	#print('display_image filename: ' + filename)
#	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug = True)
