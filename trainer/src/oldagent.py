import tensorflow as tf
import numpy as np

class Agent:

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation='tanh', input_shape=(2,), kernel_initializer='random_normal', bias_initializer='zeros'),
        ])
        # self.model.summary()
        self.params = np.concatenate([weights.flatten() for weights in self.model.get_weights()])

         # Calculate the split points for weights and biases
        self.layer_shapes = [arr.shape for arr in self.model.get_weights()]
        self.split_points = np.cumsum([np.prod(shape) for shape in self.layer_shapes[:-1]])

    def runPolicy(self, state):
        action = self.model.predict(np.array([state]), verbose=0)
        if action[0][0] >=0:
            return [1]
        else:
            return [0]
        # return [state[0] < state[1] / 2]

    def getParams(self):
        self.params = np.concatenate([weights.flatten() for weights in self.model.get_weights()])
        return self.params

    def setParams(self, params):
        # Split the flat array into weights and biases
        weights_biases = np.split(params, self.split_points)

        # Reshape weights and biases to match the shapes in the model
        reshaped_weights_biases = [arr.reshape(shape) for arr, shape in zip(weights_biases, self.layer_shapes)]

        # Set the weights and biases in the model
        self.model.set_weights(reshaped_weights_biases)