import io
import os

from flask import Flask, send_file, request, render_template

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from flask_app.mnist_nn import MnistNeuralNetwork
import numpy as np
import tensorflow as tf

app = Flask(__name__)

app.config.from_object('config.ProductionConfig')

working_folder = app.config['WORKING_FOLDER']

tf.logging.set_verbosity(tf.logging.ERROR)

datasets = input_data.read_data_sets(os.path.join(working_folder, 'data/'),  one_hot=True)

tf.logging.set_verbosity(tf.logging.INFO)

net = MnistNeuralNetwork(datasets.test, datasets.train, working_folder)


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/model/train', methods=['GET'])
def train_model():
    if net.learning:
        return 'Model is training'
    else:
        learning_rate = request.args.get('learning_rate', 0.1)
        iterations = request.args.get('iterations', 10000)
        net.training(float(learning_rate), int(iterations))
        return 'Training finished'


@app.route('/model/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_name = 'predict.jpg'
    path = os.path.join(working_folder, file_name)
    file.save(path)

    pil_im = Image.open(path)
    pix = np.array(pil_im.getdata()).reshape(pil_im.size[0], pil_im.size[1]).astype('uint8') / 255
    result = net.predict(pix.reshape(1, 784))
    return str(result)


@app.route('/model/page', methods=['GET'])
def page():
    return render_template('upload.html')


@app.route('/model/accuracy', methods=['GET'])
def compute_accuracy():
    batch_size = request.args.get('batch_size', 100)
    result = net.compute_accuracy(int(batch_size))
    return result


@app.route('/dataset/image/', methods=['GET'])
def get_image():
    id = request.args.get('id', 100)
    file_name = 'buff.jpg'
    path = os.path.join(working_folder, file_name)

    data = datasets.test.images[int(id)].reshape(28, 28) * 255
    Image.fromarray(data).convert("L").save(path)
    with open(path, 'rb') as bites:
        return send_file(
                     io.BytesIO(bites.read()),
                     attachment_filename=file_name,
                     mimetype='image/jpg'
               )

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'])

