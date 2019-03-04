import io

from flask import Flask, send_file, request, render_template

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from flask_app.mnist_nn import MnistNeuralNetwork
import numpy as np

app = Flask(__name__)

datasets = input_data.read_data_sets("data/", one_hot=True)

net = MnistNeuralNetwork(datasets.test, datasets.train)


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
    file.save(file_name)

    pil_im = Image.open(file_name)
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
    filename = 'buff.jpg'
    data = datasets.test.images[int(id)].reshape(28, 28) * 255
    Image.fromarray(data).convert("L").save(filename)
    with open(filename, 'rb') as bites:
        return send_file(
                     io.BytesIO(bites.read()),
                     attachment_filename=filename,
                     mimetype='image/jpg'
               )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')




