# from src.agent import Agent
from src.optimizer import Optimizer
from src.game_connection import GameConnection
from src.simple_agent import SimpleAgent
from src.deep_agent import DeepAgent

import time

gameUrl = "http://localhost:5000"

if (__name__ == "__main__"):

    simpleAgent = SimpleAgent(gameUrl)
    deepAgent = DeepAgent(gameUrl)
    while(True):
        simpleAgent.run()
        deepAgent.run()
        time.sleep(0.001)
        pass

    # # Create agents
    # agents = [Agent() for _ in range(10)]
    # # agents[0].setParams([-0.008, 0, 1])
    # agents[0].setParams([0, 0, 0])
    # # game.addAgents(agents)

    # initalParams = agents[0].getParams()
    # for agent in agents:
    #     agent.setParams(initalParams)

    # optimizer = Optimizer(initalParams, len(agents))

    # generation = 0
    # while True:

    #     print('Generation ', generation)

    #     for agent in agents:
    #         params, noiseIdx = optimizer.getParams()
    #         agent.setParams(params)

    #     # Rollout
    #     game.reset()
    #     while game.running:
    #         game.run(generation)

    #     # Optimize
    #     rewards = game.getScores()
    #     print(rewards)
        
    #     mr = 0
    #     m = 0
    #     for i in range(len(rewards)):
    #         if rewards[i] > mr:
    #             mr = rewards[i]
    #             r = i
    #     print(agents[m].getParams())

    #     optimizer.update(rewards)

    #     generation += 1