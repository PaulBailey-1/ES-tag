This project uses an evolutionary strategy algorithm to train a deep neural network to play a simple multiplayer tag game. 
The game environment being trained on is a web based NodeJS Express application using SocketIO. It allows players to connect and play against each other as sprites in 2D platform arena, tagging back and forth and collecting falling time coins. 

<div style="float: left;"> 
  <img src="https://github.com/PaulBailey-1/ES-tag/assets/64763623/d600b346-229b-44e7-a38d-b60a6bd8fb42" width="500px">
  <img src="https://github.com/PaulBailey-1/ES-tag/assets/64763623/3846c542-ef7c-48f0-823a-aad2d3cedafa" width="400px">
</div>
<br>
The trainer allows for the creation of multiple types of agents in order to facilitate training in steps, including a simple programmed agent, a tagger deep agent, an evader deep agent, and a full deep agent running two separate models. This allows both the tagger and evader to be trained initially against the capable simple agent, then fused and trained against itself. 
The agents create a connection to the game and interact with an action space of left, right, up, and down. The state representation includes various computed metrics from the game that vary by agent but compose around 7 integer inputs to the models. All agents simulate frame lag, to make the agent more resistant to non-ideal connections and induce more complex strategies. The deep agents make use of tensorflow to create and run sequential, fully connected, deep neural network models. 
The evolutionary strategy implementation is based off 
<i>Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari</i> [https://arxiv.org/pdf/1802.08842]
With the addition of CMA-ES implemented according to <i>Adapting Arbitrary Normal Mutation Distributions in
Evolution Strategies: The Covariance Matrix Adaptation</i> [https://ieeexplore.ieee.org/document/542381].
The trainer is designed to be run on a cloud cluster in parallel. The game server and trainer both are containerized with Docker. The trainer uses MPI to connect multiple container instances across machines in order to distribute the evaluation of each generation. The images are deployed to AWS Container Image Repositories, where they can then be used to start AWS batch job. This was used to create up to 10 C5.large EC2 instances, each with a game server and training running on a CPU core. A shared elastic file system is used to connect the MPI applications and to share logs with an instance running a TensorBoard server, for monitoring training. The policy evaluation can also be watched in real time by connecting an application load balancer to expose an instances game server. 
Currently, the training does not converge to anything, likely due to an insufficient population size. Once more parameters and algorithms have been explored, the training can easily be scaled on the cloud, although it quickly becomes expensive which has been limiting so far.

<div style="float: left">  
  <img src="https://github.com/PaulBailey-1/ES-tag/assets/64763623/000d110a-ed3d-4c89-b95e-8dc8b6a92329" width="500px">
  <img src="https://github.com/PaulBailey-1/ES-tag/assets/64763623/aa45cd2a-2bb1-41f2-8652-987a4c18f744" width="500px">
</div>
