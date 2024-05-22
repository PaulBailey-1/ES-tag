FROM tensorflow/tensorflow:latest
WORKDIR /app
RUN apt upgrade &&\
    apt update &&\
    apt install -y openmpi-bin libopenmpi-dev dos2unix
RUN pip install mpi4py python-socketio[client]

ADD ./wireup.sh /wireup
RUN dos2unix /wireup \ 
&& chmod +x /wireup \
&&  /wireup install
ADD ./entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh \
&& chmod +x /entrypoint.sh
RUN apt clean \
&&  rm -rf /var/cache/apt

COPY . .
EXPOSE 5000
ENTRYPOINT [ "/entrypoint.sh" ]
# CMD mpiexec --allow-run-as-root -n 1 python main.py "http://game-server:5000"