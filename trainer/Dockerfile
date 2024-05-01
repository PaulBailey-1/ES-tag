FROM tensorflow/tensorflow:latest AS build
WORKDIR /app
RUN apt upgrade
RUN apt update
RUN apt install -y openmpi-bin libopenmpi-dev
RUN pip install mpi4py python-socketio[client]
COPY . .

EXPOSE 5000
CMD mpiexec --allow-run-as-root -n 1 python main.py "http://game-server:5000"