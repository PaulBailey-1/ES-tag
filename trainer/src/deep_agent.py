
import numpy as np

from src.agent import Agent
from src.agent_model import AgentModel

class DeepAgent(Agent):

    def __init__(self, gameUrl, stateDim=5, config=None, modelPath=None):
        super().__init__(gameUrl, config)
        self.activeModel = AgentModel(stateDim, config=config['network'], modelPath=modelPath)

    def policy(self, agentData, playersData, powerUpsData):

        if len(playersData) > 0:
            state = self.reduceState(agentData, playersData, powerUpsData)

            rawAction = self.activeModel.predict(np.array([state]), verbose=0)[0]
            action = [bool(x > 0) for x in rawAction]
            self.conn.move(action)

    def reduceState(self, agentData, playersData, powerUpsData):
        return self.basicReduceState(agentData, playersData, powerUpsData)
    
    def basicReduceState(self, agentData, playersData, powerUpsData):

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
    def __init__(self, gameUrl, config=None, modelPath=None):
        super().__init__(gameUrl, stateDim=7, config=config, modelPath=modelPath)
        print("Created Tagger Deep Agent")

    def reduceState(self, agentData, playersData, powerUpsData):
        return self.basicReduceState(agentData, playersData, powerUpsData) + \
            self.taggerReduceState(agentData, playersData, powerUpsData)

    def taggerReduceState(self, agentData, playersData, powerUpsData):
        # Distances to closest player
        state = []
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
    def __init__(self, gameUrl, config=None, modelPath=None):
        super().__init__(gameUrl, stateDim=7, config=config, modelPath=modelPath)
        print("Created Evader Deep Agent")

    def reduceState(self, agentData, playersData, powerUpsData):
        return self.basicReduceState(agentData, playersData, powerUpsData) + \
            self.evaderReduceState(agentData, playersData, powerUpsData)

    def evaderReduceState(self, agentData, playersData, powerUpsData):
        # Distances to tagger
        state = []
        tagger = None
        for player in playersData:
            if playersData[player]['color'] == 'red':
                tagger = playersData[player]
        
        state.append(tagger['x'] - agentData['x'])
        state.append(tagger['y'] - agentData['y'])

        return state

class FullDeepAgent(TaggerDeepAgent, EvaderDeepAgent):
    def __init__(self, gameUrl, config=None, taggerModelPath=None, evaderModelPath=None):
        Agent.__init__(self, gameUrl, config)

        self.taggerModel = AgentModel(9, config=config['network'], modelPath=taggerModelPath)
        self.evaderModel = AgentModel(9, config=config['network'], modelPath=evaderModelPath)

        print("Created Full Deep Agent")

    def reduceState(self, agentData, playersData, powerUpsData):
        return self.basicReduceState(agentData, playersData, powerUpsData) + \
            self.evaderReduceState(agentData, playersData, powerUpsData)

    def policy(self, agentData, playersData, powerUpsData):

        if agentData['color'] == 'red':
            self.activeModel = self.taggerModel
        else:
            self.activeModel = self.evaderModel

        super().policy(agentData, playersData, powerUpsData)