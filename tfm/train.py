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

from tfm.methods.temporal_flow_matching_method import TemporalFlowMatching


def build_model(args: argparse.Namespace, device: torch.device, data_shape: Tuple) -> nn.Module:
    # todo: build the option to have different models here
    model = TemporalFlowMatching(
        in_shape=data_shape,
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

    for batch_idx, batch in enumerate(loader):

        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    return running_loss / max(1, num_batches)


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    device = get_device(args.device)

    os.makedirs(args.save - dir if hasattr(args, "save-dir") else args.save_dir, exist_ok=True)

    train_loader = build_dataloader(args)
    data_shape = train_loader.dataset._get_data_shape()
    model = build_model(args, device, data_shape)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Using device: {device}")
    print(f"Number of train batches: {len(train_loader)}")

    best_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # trivial "best" checkpoint
        if avg_loss < best_loss and not args.debug:
            best_loss = avg_loss
            import datetime
            from pathlib import Path
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
