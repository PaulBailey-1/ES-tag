FROM tensorflow/tensorflow:latest
WORKDIR /app
RUN apt upgrade &&\
    apt update &&\
    apt install -y openmpi-bin libopenmpi-dev
RUN pip install mpi4py python-socketio[client]

ARG WIREUP_URI=https://raw.githubusercontent.com/aws-samples/batch-mpi-examples/main/util/wireup.sh
ADD ./wireup.sh /wireup
RUN chmod +x /wireup \
&&  /wireup install
ADD ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN apt clean \
&&  rm -rf /var/cache/apt

COPY . .
EXPOSE 5000
ENTRYPOINT [ "/entrypoint.sh" ]
# CMD mpiexec --allow-run-as-root -n 1 python main.py "http://game-server:5000"