import torch
import numpy as np
import zuko

# -------------------------
# Dataset Directory
# -------------------------
dir = '/eos/user/m/mbaessle/SWAN_projects/AI_PLOTTER/datasets'

# -------------------------
# Device
# -------------------------
device_flow = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load your data tensors
# -------------------------
ztrain_scaled          = torch.load(f"{dir}/ztrain_scaled.pt")
training_conditions    = torch.load(f"{dir}/training_conditions.pt")      # or None
wtrain_scaled          = torch.load(f"{dir}/wtrain_scaled.pt")
zval_scaled            = torch.load(f"{dir}/zval_scaled.pt")
validation_conditions  = torch.load(f"{dir}/validation_conditions.pt")    # or None
wval_scaled            = torch.load(f"{dir}/wval_scaled.pt")

# -------------------------
# Dimensions
# -------------------------
z_width = ztrain_scaled.shape[1]
conditions_dim = 0 if training_conditions is None else training_conditions.shape[1]

# -------------------------
# Hyperparameters
# -------------------------
n_transforms = 2
n_splines_bins = 8
aux_nodes = 256
aux_layers = 2
n_passes_flow = 2
initial_lr = 3e-3
scheduler_patience = 5
early_stop_patience = 15
batch_size = 8192*4
weight_decay = 1e-5
max_epoch_number = 250

# --------------------------
# Parameter Printer 
# --------------------------
params = {
    "n_transforms": n_transforms,
    "n_splines_bins": n_splines_bins,
    "aux_nodes": aux_nodes,
    "aux_layers": aux_layers,
    "n_passes_flow": n_passes_flow,
    "initial_lr": initial_lr,
    "scheduler_patience": scheduler_patience,
    "early_stop_patience": early_stop_patience,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "max_epoch_number": max_epoch_number,
    "z_width": z_width,
    "conditions_dim": conditions_dim,
}


# -------------------------
# Early stopper
# -------------------------
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=early_stop_patience, min_delta=1e-2)

# -------------------------
# Flow
# -------------------------
flow = zuko.flows.NSF(
    z_width,
    context=conditions_dim if conditions_dim > 0 else None,
    bins=n_splines_bins,
    transforms=n_transforms,
    hidden_features=[aux_nodes] * aux_layers,
    passes=n_passes_flow,
).to(device_flow)

# -------------------------
# DataLoaders
# -------------------------
use_pin_memory = (device_flow.type == "cuda")

use_workers = 4 if use_pin_memory else 0

training_dataset = torch.utils.data.TensorDataset(
    ztrain_scaled, training_conditions, wtrain_scaled
)
training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=use_workers,
    pin_memory=use_pin_memory,
    persistent_workers=True if use_workers > 0 else False,
    prefetch_factor=4 if use_workers > 0 else None,
)

validation_dataset = torch.utils.data.TensorDataset(
    zval_scaled, validation_conditions, wval_scaled
)
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=min(2048, len(validation_dataset)),
    shuffle=False,
    num_workers=use_workers,
    pin_memory=use_pin_memory,
    persistent_workers=True if use_workers > 0 else False,
    prefetch_factor=4 if use_workers > 0 else None,
)

# -------------------------
# Optimizer + scheduler
# -------------------------
optimizer = torch.optim.AdamW(flow.parameters(), lr=initial_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=scheduler_patience, factor=0.3, min_lr=1e-6
)
