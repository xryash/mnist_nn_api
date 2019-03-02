import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class MnistNeuralNetwork(object):
    def __init__(self, test_data, train_data):
        self.test_data = test_data
        self.train_data = train_data
        self._learning = False

    @property
    def learning(self):
        return self._learning



    @staticmethod
    def index(array, item):
        for num, val in np.ndenumerate(array):
            if val == item:
                return num[0]


    @staticmethod
    def sigmoid(x):
        return tf.divide(
            tf.constant(1.0),
            tf.add(tf.constant(1.0), tf.exp(tf.negative(x)))
        )


    @staticmethod
    def sigmoid_derivative(x):
        return tf.multiply(
            MnistNeuralNetwork.sigmoid(x),
            tf.subtract(tf.constant(1.0), MnistNeuralNetwork.sigmoid(x))
        )


    def training(self, learning_rate, iterations):

        #flag for api
        self._learning = True

        # define gateways
        layer_0 = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])

        nodes = 40

        # initialize weights and biases
        weights_1 = tf.Variable(name='weights_1', initial_value=tf.truncated_normal([784, nodes]))
        bias_1 = tf.Variable(name='bias_1', initial_value=tf.truncated_normal([1, nodes]))

        weights_2 = tf.Variable(name='weights_2', initial_value=tf.truncated_normal([nodes, 10]))
        bias_2 = tf.Variable(name='bias_2', initial_value=tf.truncated_normal([1, 10]))

        # compute layer 1 values by sigmoid (X * W + b)
        values_1 = tf.add(tf.matmul(layer_0, weights_1), bias_1)
        layer_1 = MnistNeuralNetwork.sigmoid(values_1)

        # compute layer 2 values exactly as on layer 1
        values_2 = tf.add(tf.matmul(layer_1, weights_2), bias_2)
        layer_2 = MnistNeuralNetwork.sigmoid(values_2)

        # compute layer 2 error by  y - sigmoid (X * W + b)
        layer_2_error = tf.subtract(y, layer_2)

        # compute mean squared error
        mse = tf.losses.mean_squared_error(y, layer_2)

        # compute layer 2 delta by error * sigmoid_derivative(x) * learning_rate
        layer_2_delta = tf.multiply(
            layer_2_error,
            MnistNeuralNetwork.sigmoid_derivative(values_2)
        )

        bias_2_delta = layer_2_delta

        # compute weights 2 changes
        weights_2_delta = tf.multiply(tf.constant(learning_rate), tf.matmul(tf.transpose(layer_1), layer_2_delta))

        # compute error for layer 1 by W2 * L2_delta
        layer_1_error = tf.matmul(layer_2_delta, tf.transpose(weights_2))

        # compute layer 1 delta
        layer_1_delta = tf.multiply(
            layer_1_error,
            MnistNeuralNetwork.sigmoid_derivative(values_1)
        )

        bias_1_delta = layer_1_delta

        # compute weights 1 changes
        weights_1_delta = tf.multiply(tf.constant(learning_rate), tf.matmul(tf.transpose(layer_0), layer_1_delta))

        # optimization step
        step = [
            # update weights
            tf.assign(weights_2, tf.add(weights_2, weights_2_delta)),
            tf.assign(weights_1, tf.add(weights_1, weights_1_delta)),
            # update biases
            tf.assign(bias_2, tf.add(bias_2, tf.reduce_mean(bias_2_delta, axis=[0]))),
            tf.assign(bias_1, tf.add(bias_1, tf.reduce_mean(bias_1_delta, axis=[0])))
        ]

        # define saver for the model
        saver = tf.train.Saver()

        # start tensorflow session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for i in range(iterations):

                batch_xs, batch_ys = self.train_data.next_batch(10)
                session.run(step, feed_dict={layer_0: batch_xs, y: batch_ys})
                # print error after each thousand of elements
                if i % 1000 == 0:
                    error = session.run(mse, feed_dict={
                        layer_0: self.test_data.images[:1000],
                        y: self.test_data.labels[:1000]})
                    print(error)

            print('Test')
            print(session.run(mse, feed_dict={
                layer_0: self.test_data.images[:10000],
                y: self.test_data.labels[:10000]}))

            #save model
            save_path = saver.save(session, "tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

        session.close()
        self._learning = False

    def predict(self, images):

        layer_0 = tf.placeholder(tf.float32, [None, 784])

        nodes = 40

        weights_1 = tf.get_variable('weights_1', shape=[784, nodes])
        bias_1 = tf.get_variable('bias_1', shape=[1, nodes])

        weights_2 = tf.get_variable('weights_2', shape=[nodes, 10])
        bias_2 = tf.get_variable('bias_2', shape=[1, 10])

        values_1 = tf.add(tf.matmul(layer_0, weights_1), bias_1)
        layer_1 = MnistNeuralNetwork.sigmoid(values_1)

        values_2 = tf.add(tf.matmul(layer_1, weights_2), bias_2)
        layer_2 = MnistNeuralNetwork.sigmoid(values_2)

        saver = tf.train.Saver()

        predicts = []

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            saver.restore(session, "tmp/model.ckpt")
            print("Model loaded")

            print('Predicts')
            for image in images:
                result = session.run(layer_2, feed_dict={layer_0: image.reshape(1, 784)})
                #plt.imshow(image.reshape(28, 28))
                #plt.show()
                i = zip(range(10), result[0])
                #[print(j) for j in i]

                max_value = max(result[0])

                pred = MnistNeuralNetwork.index(result[0], max_value)

                print((max_value, pred))
                print()
                predicts.append([max_value, pred])

        session.close()
        return predicts

    def compute_accuracy(self, batch_size):
        images_batch = self.test_data.images[:batch_size]
        labels_batch = self.test_data.labels[:batch_size]
        predicts = self.predict(images_batch)
        accuracy = 0
        for i in range(len(predicts)):
            predict_class = predicts[i][1]
            label_class = MnistNeuralNetwork.index(labels_batch[i], 1)
            if predict_class is label_class:
                    accuracy += 1
        result = '{0} / {1}'.format( accuracy, len(predicts))
        print(result)
        percents = accuracy / len(predicts) * 100
        print('{}%'.format(percents))
        return '{0}, accuracy = {1}'.format(result, percents)
