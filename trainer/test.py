from src.simple_agent import SimpleAgent
from src.deep_agent import FullDeepAgent

import time
import numpy as np

gameUrl = "http://localhost:5000"
testTime = 10

if (__name__ == "__main__"):

    # Create agents
    deepAgent = FullDeepAgent(gameUrl)

    while(True):
        deepAgent.run()
        time.sleep(0.001)