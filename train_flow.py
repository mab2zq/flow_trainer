#!/usr/bin/env python3
# Human readable block

import os
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.amp import autocast

from model_setup import (
    flow,
    optimizer,
    scheduler,
    early_stopper,
    training_dataloader,
    validation_dataloader,
    device_flow,
    max_epoch_number,
    params,
)

# Global perf flags
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

accum_steps = 4

cluster = os.environ.get("CONDOR_CLUSTER")
proc = os.environ.get("CONDOR_PROC")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-dir",
    type=str,
    default=None,
    help="Optional directory for saving checkpoints",
)
args = parser.parse_args()

# Default behavior (original)
default_save_dir = f"/afs/cern.ch/user/m/mbaessle/flow_trainer/checkpoints/"

# If user passes --run-dir, override
save_dir = args.run_dir if args.run_dir is not None else default_save_dir

Path(save_dir).mkdir(parents=True, exist_ok=True)

training_loss_array = []
validation_loss_array = []


def train_step(inputs, conditions, weights):
    with autocast(device_flow.type, enabled=False):
        joint_dist = flow(conditions)
        joint_logp = joint_dist.log_prob(inputs)

    neg_logp = -joint_logp
    numerator = (weights * neg_logp).sum()
    denominator = weights.sum().clamp_min(1e-12)
    loss = numerator / denominator
    return loss, numerator, denominator


def main():
    print("# -------------------------")
    print(f"Cluster: {cluster}   Process: {proc}")
    print("# -------------------------")
    print("# Hyperparameters")
    print("# -------------------------")
    for k, v in params.items():
        print(f"{k:20s}: {v}")

    best_val = float("inf")
    best_epoch = -1
    best_flow = flow.state_dict()
    best_optimizer = optimizer.state_dict()

    for epoch in range(max_epoch_number):
        t0 = time.time()
        flow.train()
        train_num_t = 0.0
        train_den_t = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, conditions, weights) in enumerate(training_dataloader):
            inputs = inputs.to(device_flow, non_blocking=True)
            weights = weights.to(device_flow, non_blocking=True)
            ctx = conditions.to(device_flow, non_blocking=True) if conditions is not None else None

            loss, numerator, denominator = train_step(inputs, ctx, weights)

            (loss / accum_steps).backward()

            if ((batch_idx + 1) % accum_steps == 0) or (batch_idx == len(training_dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_num_t += numerator.detach().cpu().item()
            train_den_t += denominator.detach().cpu().item()

        train_loss = train_num_t / (train_den_t + 1e-12)
        training_loss_array.append(train_loss)

        # Validation
        flow.eval()
        val_num_t = 0.0
        val_den_t = 0.0
        with torch.no_grad():
            for v_inputs, v_conditions, v_weights in validation_dataloader:
                v_inputs = v_inputs.to(device_flow, non_blocking=True)
                v_weights = v_weights.to(device_flow, non_blocking=True)
                v_ctx = v_conditions.to(device_flow, non_blocking=True) if v_conditions is not None else None

                with autocast(device_flow.type, enabled=False):
                    joint_dist_v = flow(v_ctx)
                    joint_logp_v = joint_dist_v.log_prob(v_inputs)

                neg_logp_v = -joint_logp_v
                numerator_v = (v_weights * neg_logp_v).sum()
                denominator_v = v_weights.sum().clamp_min(1e-12)

                val_num_t += numerator_v.detach().cpu().item()
                val_den_t += denominator_v.detach().cpu().item()

        val_loss = val_num_t / (val_den_t + 1e-12)
        validation_loss_array.append(val_loss)

        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_flow = flow.state_dict()
            best_optimizer = optimizer.state_dict()

        print(
            f"Epoch {epoch:03d} | Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | epoch time: {time.time() - t0:.2f}s",
            flush=True,
        )

        if early_stopper.early_stop(val_loss) or epoch >= max_epoch_number - 1:
            print(f"Stopping. Best val: {best_val:.6f} at epoch {best_epoch}", flush=True)
            torch.save(
                {
                    "flow_state": best_flow,
                    "optimizer": best_optimizer,
                    "epoch": best_epoch,
                    "training_loss_array": training_loss_array,
                    "validation_loss_array": validation_loss_array,
                },
                os.path.join(save_dir, f"best_model_{cluster}.pth"),
            )
            break


if __name__ == "__main__":
    main()
