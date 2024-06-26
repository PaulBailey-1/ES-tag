from src.logger import log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import numpy as np

class AgentModel():
    def __init__(self, inputDim, config=None, modelPath=None):

        log(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        if modelPath:
            self.model = tf.keras.models.load_model(modelPath)
        else:
            # Build neural network model
            hiddenLayers = 2
            hiddenLayerDim = 10 
            hiddenNonlin =  "relu"
            initialSigma =  0.05
            if (config):
                if "hiddenLayers" in config: hiddenLayers = config['hiddenLayers']
                if "hiddenLayerDim" in config: hiddenLayerDim = config['hiddenLayerDim']
                if "hiddenNonlin" in config: hiddenNonlin = config['hiddenNonlin']
                if "initialSigma" in config: initialSigma = config['initialSigma']

            kernelInit = tf.keras.initializers.RandomNormal(mean=0.0, stddev=initialSigma)
            self.model = tf.keras.Sequential([
                tf.keras.Input(shape=(inputDim,))] +
                [tf.keras.layers.Dense(units=hiddenLayerDim, activation=hiddenNonlin, kernel_initializer=kernelInit, bias_initializer='zeros') for _ in range(hiddenLayers)] + [
                tf.keras.layers.Dense(units=4, activation='tanh', kernel_initializer=kernelInit, bias_initializer=kernelInit),
            ])

        self.params = np.concatenate([weights.flatten() for weights in self.model.get_weights()])

         # Calculate the split points for weights and biases
        self.layer_shapes = [arr.shape for arr in self.model.get_weights()]
        self.split_points = np.cumsum([np.prod(shape) for shape in self.layer_shapes[:-1]])

    def getParamCount(self):
        return self.model.count_params()

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

    def save(self, dir):
        self.model.save(dir + '/deepAgentModel.keras')