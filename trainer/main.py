from src.optimizer import Optimizer
from src.simple_agent import SimpleAgent
from src.deep_agent import DeepAgent, TaggerDeepAgent, EvaderDeepAgent

import time
import sys
import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)
rank = comm.Get_rank()
workersCount = comm.Get_size()

gameUrl = "http://localhost:5000"
testTime = 5

logfile = None
def log(msg):
    logFile = open("log.txt", "a")
    logFile.write(msg + "\n")
    logFile.close()

if (__name__ == "__main__"):

    envGameUrl = os.environ.get('GAME_SERVER_URL')
    if envGameUrl != None:
        gameUrl = envGameUrl

    if len(sys.argv) > 1:
        gameUrl = sys.argv[1]

    gameUrl = gameUrl.replace("RANK", str(rank))

    print(f"Started process {rank} on {MPI.Get_processor_name()}")

    paramCount = None

    # Create agents
    deepAgent = TaggerDeepAgent(gameUrl)
    simpleAgent = SimpleAgent(gameUrl)
    paramCount = deepAgent.getParamCount()

    print(f"Process {rank}: created agents")
    
    if rank == 0: # root

        log(f"Beginning training with {workersCount} workers")
        
        # Set the starting parameters
        optimizer = Optimizer(deepAgent.getParams(), workersCount)
        print("Initialized optimizer")

    while True:

        # Rollout new params
        paramSendBuf = None
        if rank == 0:

            paramSendBuf = np.empty([workersCount, paramCount], dtype='f')
            paramSendBuf[:] = [optimizer.getParams() for _ in range(workersCount)]

        paramRecvBuf = np.empty(paramCount, dtype='f')
        comm.Scatter(paramSendBuf, paramRecvBuf, root=0)

        # Run agents
        deepAgent.reset()
        simpleAgent.reset()
        
        deepAgent.conn.restartGame()
        deepAgent.conn.startGame(True)

        while(True):
            simpleAgent.run()
            deepAgent.run()

            if deepAgent.score != -1:
                if (not deepAgent.isRed or deepAgent.score <= 60 - testTime):
                    break
            
            time.sleep(0.001)

        rewardSendbuf = np.empty([1], dtype='i')
        rewardSendbuf[0] = deepAgent.score

        rewardRecvBuf = None
        if rank == 0:
            rewardRecvBuf = np.empty([workersCount], dtype="i")
        
        comm.Gather(rewardSendbuf, rewardRecvBuf, root=0)

        if rank == 0:
            optimizer.update(rewardRecvBuf)
