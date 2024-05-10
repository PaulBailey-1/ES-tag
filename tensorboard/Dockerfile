FROM python:3.8.0-slim

RUN pip install --upgrade pip
RUN pip install tensorboard six && pip uninstall -y protobuf && pip install protobuf==4.25

EXPOSE 6006
CMD tensorboard --logdir /mount/efs/logs/ --host 0.0.0.0 --port 6006