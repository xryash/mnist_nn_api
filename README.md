# mnist_nn_api

Neural network works with mnist images.

# API

GET
/model/page - to get uploading images page and prediction.

GET
/model/train - to start neural network training.
Args:
learning rate - training speed, 0.1 by default.
iterations - iterations number, 10 000 by default.

GET
/model/accuracy - to get model accuracy
Args:
batch_size - images list size for computing accuracy, 100 by default.

GET
/dataset/image/ - to get
Args:
id - image number, 100 by default.
