from src.optimizer import Optimizer
from src.simple_agent import SimpleAgent
from src.deep_agent import DeepAgent, TaggerDeepAgent, EvaderDeepAgent

import time
import numpy as np

gameUrl = "http://localhost:5000"
testTime = 10

if (__name__ == "__main__"):

    # # Create agents
    # simpleAgent = SimpleAgent(gameUrl)
    # deepAgent = TaggerDeepAgent(gameUrl)

    # # Run agents
    # deepAgent.conn.restartGame()
    # deepAgent.conn.startGame(True)

    # while(True):
    #     simpleAgent.run()
    #     deepAgent.run()
    #     time.sleep(0.001)
    #     print(deepAgent.isRed)

    f = open("log.txt", "w")
    while (True):
        f.write("hi there\n")
        time.sleep(1)
    # f.close()