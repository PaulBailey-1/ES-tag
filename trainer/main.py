from src.optimizer import Optimizer
from src.simple_agent import SimpleAgent
from src.deep_agent import DeepAgent, TaggerDeepAgent, EvaderDeepAgent

import time
import datetime
import numpy as np
from mpi4py import MPI
from argparse import ArgumentParser

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)
rank = comm.Get_rank()
workersCount = comm.Get_size()

testTime = 5

def main(gameUrl, config, runName, modelPath, agentType):

    # gameUrl = gameUrl.replace("RANK", str(rank))
    paramCount = None
    if config:
        testTime = config['rl']['testTime']
    logDir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if runName:
        logDir = runName + '-' + logDir

    print(f"Started trainer process {rank} on {MPI.Get_processor_name()}")

    # Create agents
    agentType = TaggerDeepAgent
    if agentType == 'evader':
        agentType = EvaderDeepAgent

    deepAgent = agentType(gameUrl, config=config['agent'], modelPath=modelPath)
    simpleAgent = SimpleAgent(gameUrl, config=config['agent'])
    paramCount = deepAgent.getParamCount()

    print(f"Process {rank}: created agents")
    
    if rank == 0: # root

        print(f"Beginning training with {workersCount} workers")
        generation = 0
        # Set the starting parameters
        optimizer = Optimizer(deepAgent.getParams(), workersCount, logDir, config=config['optimizer'])

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
                exCon = deepAgent.isRed
                if agentType == 'tagger': exCon = not exCon
                if (exCon or deepAgent.score <= 60 - testTime):
                    break
            
            time.sleep(0.001)

        rewardSendbuf = np.empty([1], dtype='i')
        rewardSendbuf[0] = deepAgent.score

        rewardRecvBuf = None
        if rank == 0:
            rewardRecvBuf = np.empty([workersCount], dtype="i")
        
        comm.Gather(rewardSendbuf, rewardRecvBuf, root=0)

        if rank == 0:
            print("Generation ", generation)
            optimizer.update(generation, rewardRecvBuf)

            if generation % 10 == 0:
                deepAgent.save(logDir)
            generation += 1

if (__name__ == "__main__"):

    parser = ArgumentParser()
    parser.add_argument('-g', '--game_url', help="URL of game server", default="http://localhost:5000", type=str)
    parser.add_argument('-c', '--config_file', help='Path to configuration file', type=str)
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    parser.add_argument('-m', '--model_path', help='Path to keras model to use for a deep agent', type=str)
    parser.add_argument('-a', '--agent_type', help='Type of agent to train (tagger, evader, full)', type=str, default="tagger")
    args = parser.parse_args()

    main(args.game_url, args.config_file, args.run_name, args.model_path, args.agent_type)