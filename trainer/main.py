from src.optimizer import Optimizer
from src.simple_agent import SimpleAgent
from src.deep_agent import DeepAgent, TaggerDeepAgent, EvaderDeepAgent

import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)
rank = comm.Get_rank()
workersCount = comm.Get_size()

gameUrl = "http://localhost:500" + str(rank)
testTime = 10

logfile = None
def log(msg):
    logFile = open("log.txt", "a")
    logFile.write(msg + "\n")
    logFile.close()

if (__name__ == "__main__"):

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
        generation = 0
        print("Initilized optimizer")

    while True:

        # Rollout new params
        paramSendBuf = None
        if rank == 0:

            # print('Generation ', generation)
            log(f"Generation {str(generation)}")

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

        reward = deepAgent.score - 60 + testTime
        rewardSendbuf = np.empty([1], dtype='i')
        rewardSendbuf[0] = reward

        rewardRecvBuf = None
        if rank == 0:
            rewardRecvBuf = np.empty([workersCount], dtype="i")
        
        comm.Gather(rewardSendbuf, rewardRecvBuf, root=0)

        if rank == 0:
            
            # Optimize
            rewards = rewardRecvBuf
            # print(rewards)
            log(f"Rewards: {rewards}")
            
            mr = 0
            m = 0
            for i in range(len(rewards)):
                if rewards[i] > mr:
                    mr = rewards[i]
                    r = i

            optimizer.update(rewards)

            generation += 1
