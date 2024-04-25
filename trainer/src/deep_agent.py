import tensorflow as tf
import numpy as np

from src.agent import Agent

class DeepAgent(Agent):

    def __init__(self, gameUrl, stateDim=5):
        super().__init__(gameUrl)

        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(stateDim,)),
            tf.keras.layers.Dense(units=10, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'),
            tf.keras.layers.Dense(units=10, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'),
            tf.keras.layers.Dense(units=4, activation='tanh', kernel_initializer='random_normal', bias_initializer='zeros'),
        ])
        # self.model.summary()

        self.params = np.concatenate([weights.flatten() for weights in self.model.get_weights()])

         # Calculate the split points for weights and biases
        self.layer_shapes = [arr.shape for arr in self.model.get_weights()]
        self.split_points = np.cumsum([np.prod(shape) for shape in self.layer_shapes[:-1]])

    def policy(self, agentData, playersData, powerUpsData):

        if len(playersData) > 0:

            state = self.reduceState(agentData, playersData, powerUpsData)
            # state = [x / 1000 for x in state]
            # print(state)

            rawAction = self.model.predict(np.array([state]), verbose=0)[0]
            # print(rawAction)
            action = [bool(x > 0) for x in rawAction]
            self.conn.move(action)

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
    
    def reduceState(self, agentData, playersData, powerUpsData):

        # Distance to walls and floor
        state = []
        
        state.append(agentData['x'])
        state.append(1000 - agentData['x'])
        state.append(500 - agentData['y'])

        # Distances to powerup
        if len(powerUpsData) > 0:
            powerUp = powerUpsData[0]
            state.append(powerUp['x'] - agentData['x'])
            state.append(powerUp['y'] - agentData['y'])
        else:
            state.append(0) # This might work ...
            state.append(0)

        return state

class TaggerDeepAgent(DeepAgent):
    def __init__(self, gameUrl):
        super().__init__(gameUrl, 7)

    def reduceState(self, agentData, playersData, powerUpsData):
        state = super().reduceState(agentData, playersData, powerUpsData)

        # Distances to closest player
        leastXDist = 10000
        closestPlayer = None
        for playerId in playersData:
            dist = abs(playersData[playerId]['x'] - agentData['x'])
            if (leastXDist > dist):
                leastXDist = dist
                closestPlayer = playersData[playerId]
        
        state.append(closestPlayer['x'] - agentData['x'])
        state.append(closestPlayer['y'] - agentData['y'])

        return state

class EvaderDeepAgent(DeepAgent):
    def __init__(self, gameUrl):
        super().__init__(gameUrl, 7)

    def reduceState(self, agentData, playersData, powerUpsData):
        state = super().reduceState(agentData, playersData, powerUpsData)

        # Distances to tagger
        tagger = None
        for player in playersData:
            if playersData[player]['color'] == 'red':
                tagger = playersData[player]
        
        state.append(tagger['x'] - agentData['x'])
        state.append(tagger['y'] - agentData['y'])

        return state
