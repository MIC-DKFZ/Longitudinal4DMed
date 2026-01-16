#!/usr/bin/env python
import argparse
import os
import random
from typing import Tuple

from utils.parser import get_args
from utils.util_functions import *
from data_loaders.data_utils import build_dataloader
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


import datetime
from pathlib import Path
from utils.validation_utils import val_step
from tfm.methods.temporal_flow_matching_method import TemporalFlowMatching
from tfm.methods.cronos import CRONOS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tqdm

def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    # todo: build the option to have different models here
    model_type = 'cronos' # args.model_type
    if model_type == 'cronos':
        model = CRONOS(
            feature_size=args.base_channels,
            **(vars(args)),
        )
    else:
        print('No valid model used: Defaulting to Temporal Flow Matching model')
        model = TemporalFlowMatching(
            feature_size=args.base_channels,
            **(vars(args)),
        )
    model.to(device)
    return model


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        log_interval: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    pbar = tqdm.tqdm(loader, desc=f"Epoch: {epoch}", leave=True, ncols=130)

    for batch_idx, batch in enumerate(pbar):

        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


def main() -> None:
    args = get_args()
    if args.debug:
        print("Running in debug mode.")
        args.num_epochs = 2
        args.log_interval = 1
    set_seed(args.seed)
    device = get_device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = build_dataloader(args)
    validation_loader = build_dataloader(args,train_test_val='val')
    data_shape = train_loader.dataset._get_data_shape()
    args.in_shape = data_shape
    model = build_model(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_linear_warmup_scheduler(optimizer, num_epochs=args.num_epochs)
    print(f"Using device: {device}")
    print(f"Number of train batches: {len(train_loader)}")

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        scheduler.step()
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        if epoch % args.log_interval == 0:
            val_result = val_step(validation_loader, model, min_val=best_val, **vars(args))
            # currently, we still insert the best loss into the val step, will be deprecated
            avg_val = val_result[1]

            # "best" checkpoint
            if avg_val < best_val and not args.debug:
                best_val = avg_val
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ckpt_path = Path(args.save_dir) / f"{current_time}_tfm_best.pt"
                # ckpt_path = os.path.join(args.save_dir, f"{current_time}_tfm_best.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
