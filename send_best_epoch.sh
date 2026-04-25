#!/bin/bash

#run with ./send_best_epoch.s <subdirectory_in_checkpoints>

cd ~/flow_trainer

shopt -s nullglob
maxnum=-1
maxfile=""
for f in checkpoints/$1/*.pth; do
  if [[ $f =~ ([0-9]+)\.pth$ ]]; then
    n=${BASH_REMATCH[1]}
    (( n > maxnum )) && { maxnum=$n; maxfile=$f; }
  fi
done
printf '%s\n' "$maxfile"
cp $maxfile /eos/user/m/mbaessle/SWAN_projects/AI_PLOTTER/lxplus_ckpts
