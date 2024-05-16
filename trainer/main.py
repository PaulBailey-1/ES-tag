from src.optimizer import Optimizer
from src.simple_agent import SimpleAgent
from src.deep_agent import TaggerDeepAgent, EvaderDeepAgent, FullDeepAgent

import time
import datetime
import json
import numpy as np
from mpi4py import MPI
from argparse import ArgumentParser

comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)
rank = comm.Get_rank()
workersCount = comm.Get_size()


def log(str):
    # f = open("log.txt", "a")
    # f.write(str + "\n")
    # f.close()
    print(f"{rank}: {str}")

def main(gameUrl, config, runName, logPath, modelPath, secondModelPath, agentTypeS):

    gameUrl = gameUrl.replace("RANK", str(rank))
    paramCount = None
    testTime = 5
    if config:
        testTime = config['rl']['testTime']
    logDir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if runName:
        logDir = runName + '-' + logDir
    logDir = logPath + logDir

    log(f"Started trainer process on {MPI.Get_processor_name()}")

    # Create agents
    agentType = TaggerDeepAgent
    if agentTypeS == 'evader':
        agentType = EvaderDeepAgent
    elif agentTypeS == 'full':
        agentType = FullDeepAgent

    agentConfig = None
    if config:
        agentConfig = config['agent']

    if agentType == FullDeepAgent:
        deepAgent = agentType(gameUrl, agentConfig, taggerModelPath=modelPath, evaderModelPath=secondModelPath)
    else:
        deepAgent = agentType(gameUrl, agentConfig, modelPath=modelPath)

    simpleAgent = SimpleAgent(gameUrl, agentConfig)
    paramCount = deepAgent.activeModel.getParamCount()

    log("Created agents")
    
    if rank == 0: # root

        log(f"Beginning training with {workersCount} workers")
        generation = 0
        # Set the starting parameters
        optimizerConfig = None
        if config: optimizerConfig = config['optimizer']
        optimizer = Optimizer(deepAgent.activeModel.getParams(), workersCount, logDir, config=optimizerConfig)

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

        if agentType == TaggerDeepAgent:
            deepAgent.conn.startGame(True)
        elif agentType == EvaderDeepAgent:
            simpleAgent.conn.startGame(True)
        else:
            deepAgent.conn.startGame()

        avgDistance = 0
        runCount = 0
        while(True):
            if agentType == TaggerDeepAgent:
                simpleAgent.run(forceRed=False)
                deepAgent.run(forceRed=True)

            if deepAgent.score != -1:

                if agentType == TaggerDeepAgent:
                    if (deepAgent.score + simpleAgent.score < 120):
                        avgDistance += np.sqrt(pow(deepAgent.x - simpleAgent.x, 2) + pow(deepAgent.y - simpleAgent.y, 2))
                        runCount += 1

                if (deepAgent.score + simpleAgent.score <= 120 - testTime): break
                    
            time.sleep(0.001)

        rewardSendbuf = np.empty([1], dtype='i')
        rewardSendbuf[0] = -avgDistance / runCount

        rewardRecvBuf = None
        if rank == 0:
            rewardRecvBuf = np.empty([workersCount], dtype="i")
        
        comm.Gather(rewardSendbuf, rewardRecvBuf, root=0)

        if rank == 0:
            if (generation % 10 == 0):
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
    parser.add_argument('-l', '--log_path', help='Path to store logs', type=str, default="logs/")
    parser.add_argument('-m', '--model_path', help='Path to keras model to use for a deep agent', type=str)
    parser.add_argument('-s', '--second_model_path', help='Path to second keras model to use for a deep agent', type=str)
    parser.add_argument('-a', '--agent_type', help='Type of agent to train (tagger, evader, full)', type=str, default="tagger")
    args = parser.parse_args()

    config = None
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.loads(f.read())
            print("Read config from ", args.config_file)

    main(args.game_url, config, args.run_name, args.log_path, args.model_path, args.second_model_path, args.agent_type)