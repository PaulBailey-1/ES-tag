
import numpy as np

from src.agent import Agent
from src.agent_model import AgentModel

class DeepAgent(Agent):

    def __init__(self, gameUrl, stateDim=5, config=None, modelPath=None, gameTag=None):
        super().__init__(gameUrl, config, gameTag=gameTag)
        networkConfig = None
        if config and 'network' in config: networkConfig = config['network'] 
        self.activeModel = AgentModel(stateDim, config=networkConfig, modelPath=modelPath)

    def policy(self, agentData, playersData, powerUpsData):

        taggerData = None
        for player in playersData:
            if playersData[player]['color'] == 'red':
                taggerData = playersData[player]
        if agentData['color'] == 'red':
                taggerData = agentData

        if len(playersData) > 0 and taggerData != None:
            state = self.reduceState(agentData, taggerData, playersData, powerUpsData)
            rawAction = self.activeModel.model.predict(np.array([state]), verbose=0)[0]
            action = [bool(x > 0) for x in rawAction]
            self.conn.move(action)

    def reduceState(self, agentData, taggerData, playersData, powerUpsData):
        return self.basicReduceState(agentData, taggerData, playersData, powerUpsData)
    
    def basicReduceState(self, agentData, taggerData, playersData, powerUpsData):

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

    def save(self, dir):
        self.activeModel.model.save(dir + '/deepAgentModel.keras')

class TaggerDeepAgent(DeepAgent):
    def __init__(self, gameUrl, config=None, modelPath=None, gameTag=None):
        super().__init__(gameUrl, stateDim=7, config=config, modelPath=modelPath, gameTag=gameTag)
        print("Created Tagger Deep Agent")

    def reduceState(self, agentData, taggerData, playersData, powerUpsData):
        return self.basicReduceState(agentData, taggerData, playersData, powerUpsData) + \
            self.taggerReduceState(agentData, taggerData, playersData, powerUpsData)

    def taggerReduceState(self, agentData, taggerData, playersData, powerUpsData):
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
    
    def save(self, dir):
        self.activeModel.model.save(dir + '/taggerDeepAgentModel.keras')

class EvaderDeepAgent(DeepAgent):
    def __init__(self, gameUrl, config=None, modelPath=None, gameTag=None):
        super().__init__(gameUrl, stateDim=7, config=config, modelPath=modelPath, gameTag=gameTag)
        print("Created Evader Deep Agent")

    def reduceState(self, agentData, taggerData, playersData, powerUpsData):
        return self.basicReduceState(agentData, taggerData, playersData, powerUpsData) + \
            self.evaderReduceState(agentData, taggerData, playersData, powerUpsData)

    def evaderReduceState(self, agentData, taggerData, playersData, powerUpsData):
        # Distances to tagger
        state = []
        state.append(taggerData['x'] - agentData['x'])
        state.append(taggerData['y'] - agentData['y'])

        return state
    
    def save(self, dir):
        self.activeModel.model.save(dir + '/evaderDeepAgentModel.keras')

class FullDeepAgent(TaggerDeepAgent, EvaderDeepAgent):
    def __init__(self, gameUrl, config=None, taggerModelPath=None, evaderModelPath=None, gameTag=None):
        Agent.__init__(self, gameUrl, config, gameTag=gameTag)
        
        modelConfig = None
        if config: modelConfig = config['network']

        self.taggerModel = AgentModel(7, config=modelConfig, modelPath=taggerModelPath)
        self.evaderModel = AgentModel(7, config=modelConfig, modelPath=evaderModelPath)

        print("Created Full Deep Agent")

    def reduceState(self, agentData, taggerData, playersData, powerUpsData):
        state = self.basicReduceState(agentData, taggerData, playersData, powerUpsData)
        if self.activeModel == self.taggerModel:
            state += self.taggerReduceState(agentData, taggerData, playersData, powerUpsData)
        else:
            state += self.evaderReduceState(agentData, taggerData, playersData, powerUpsData)
        return state

    def policy(self, agentData, playersData, powerUpsData):

        if agentData['color'] == 'red':
            self.activeModel = self.taggerModel
        else:
            self.activeModel = self.evaderModel

        super().policy(agentData, playersData, powerUpsData)

    def save(self, dir):
        self.taggerModel.model.save(dir + '/taggerDeepAgentModel.keras')
        self.evaderModel.model.save(dir + '/evaderDeepAgentModel.keras')