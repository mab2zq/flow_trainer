#!/bin/bash

# Load LCG CUDA environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/afs/cern.ch/user/m/mbaessle/flow_trainer/zuko_dump:$PYTHONPATH

# Change directory
cd /afs/cern.ch/user/m/mbaessler/flow_trainer

# Run training
python3 train_flow.py
