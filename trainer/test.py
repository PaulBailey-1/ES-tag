from src.simple_agent import SimpleAgent

import time
import numpy as np

gameUrl = "http://localhost:5000"
testTime = 10

if (__name__ == "__main__"):

    # Create agents
    simpleAgents = [SimpleAgent(gameUrl) for _ in range(5)]

    while(True):
        for simpleAgent in simpleAgents : simpleAgent.run()
        time.sleep(0.001)