#!/bin/bash

# This sample, non-production-ready script to establish a container entrypoint for MPI Batch jobs.
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Authors: J. Lowell Wofford <jlowellw@amazon.com>

source /etc/profile
export PATH=$PATH:/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export WORKERS_PER_NODE=1

/wireup run \
  mpiexec  \
  --mca btl_tcp_if_exclude lo,ecs-eth0 \
  -hostfile /hostfile-ompi \
  -n $(($AWS_BATCH_JOB_NUM_NODES * $WORKERS_PER_NODE)) \
  python main.py -g http://game-server-RANK:5000 -l /app/logs/ -c taggerConfig.json
  # python main.py -g http://localhost:5000 -l /mount/efs/logs/ -c taggerConfig.json

# CMD mpiexec --allow-run-as-root -n 1 python main.py "http://game-server:5000"