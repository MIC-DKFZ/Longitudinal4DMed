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
from utils.validation_utils import val_step, _extract_batch, _forward_and_reshape, get_last_context_image_baseline
from src.methods.temporal_flow_matching_method import TemporalFlowMatching
from src.methods.cronos import CRONOS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tqdm
from torch.utils.tensorboard import SummaryWriter

def _log_image_grid(writer: "SummaryWriter", model, loader, device, in_shape, epoch: int) -> None:
    """Log a GT / LCI / Prediction side-by-side image to TensorBoard."""
    import torch.nn.functional as F
    try:
        batch = next(iter(loader))
        with torch.no_grad():
            batch_x, batch_y, _, _, time_points = _extract_batch(batch, device)
            pred, batch_y = _forward_and_reshape(model, batch_x, batch_y, time_points, in_shape=in_shape)
            lci = get_last_context_image_baseline(batch_x)

        def _to_slice(vol):
            # vol: (C, D, H, W) or similar -> (1, H, W) central slice, normalised
            arr = vol.detach().cpu().float()
            while arr.ndim > 3:
                arr = arr[arr.shape[0] // 2]
            if arr.ndim == 2:
                arr = arr.unsqueeze(0)
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        gt_s   = _to_slice(batch_y[0])
        lci_s  = _to_slice(lci[0])
        pred_s = _to_slice(pred[0])
        # side-by-side: (1, H, W*3)
        grid = torch.cat([gt_s, lci_s, pred_s], dim=-1)
        writer.add_image("Val/GT_LCI_Pred", grid, global_step=epoch)
    except Exception:
        pass  # image logging is best-effort; never break training


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    # todo: build the option to have different models here
    model_type = args.model_type
    # model_type = 'cronos' # args.model_type
    if model_type == 'cronos':
        model = CRONOS(
            feature_size=args.base_channels,
            **(vars(args)),
        )
    elif model_type == 'tfm':
        model = TemporalFlowMatching(
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.model_type}_{args.dataset}"
        f"_ch{args.base_channels}"
        f"_lr{args.lr}"
        f"_sig{args.training_noise}"
        f"_seed{args.seed}"
        f"_{timestamp}"
    )
    log_dir = args.log_dir if args.log_dir else os.path.join(args.save_dir, "logs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", "\n".join(f"    {k}: {v}" for k, v in sorted(vars(args).items())), 0)

    train_loader = build_dataloader(args)
    validation_loader = build_dataloader(args, train_test_val='val', shuffle=False)
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
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % args.log_interval == 0:
            val_result = val_step(validation_loader, model, min_val=best_val, **vars(args))
            # currently, we still insert the best loss into the val step, will be deprecated
            avg_val = val_result[1]
            for metric_name, metric_value in val_result[0].items():
                writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
            _log_image_grid(writer, model, validation_loader, device, data_shape, epoch)

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

    writer.close()


if __name__ == "__main__":
    main()
