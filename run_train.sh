#!/bin/bash

# Load LCG CUDA environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="/eos/user/m/mbaessle/.local/lib/python3.13/site-packages:${PYTHONPATH:-}"

# Change directory
cd ./

# Run training
python3 train_flow.py
