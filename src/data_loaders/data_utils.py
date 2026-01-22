import argparse
import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

from .acdc_loader import ACDCDataset
from .isles_loader import ISLESDataset




class DummyTemporalDataset(Dataset):
    """
    Minimal toy dataset so the script runs out of the box.

    Replace this with your own Dataset that returns:
        x:     [T, C, D, H, W]
        times: [T]
    """

    def __init__(
            self,
            length: int = 8,
            T: int = 4,
            C: int = 1,
            D: int = 32,
            H: int = 32,
            W: int = 32,
    ) -> None:
        super().__init__()
        self.length = length
        self.T = T
        self.C = C
        self.D = D
        self.H = H
        self.W = W

        # fixed normalized times for the toy example
        self._times = torch.linspace(0.0, 1.0, T)

    def __len__(self) -> int:
        return self.length


    def __getitem__(self, idx):
        # synthetic sequence: (T, C, D, H, W)
        x = torch.randn(self.T+1, self.C, self.D, self.H, self.W).clip(0, 1)

        return {
            "target_img": x[[-1]],  # (1, C, D, H, W)
            "context": x[:-1],  # (T-1, C, D, H, W)
            "target_seg": torch.zeros_like(x[[-1]]),
            "context_seg": torch.zeros_like(x[:-1]),
            "target_time": torch.tensor([1.0], dtype=torch.float32),
            "context_time": torch.linspace(0.0, 1.0, self.T+1)[:-1],
        }

    def _get_data_shape(self) -> Tuple[int, int, int, int, int]:
        return (self.T, self.C, self.D, self.H, self.W)

def build_dataloader(args: argparse.Namespace, train_test_val='trn') -> DataLoader:
    if args.dummy:
        dataset = DummyTemporalDataset()
    else:
        if args.dataset == 'acdc':
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = ACDCDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'isles':
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = ISLESDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'lumiere':
            from .lumiere_loader import LumiereDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = LumiereDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'oasis':
            from .oasis_loader import OASISDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = OASISDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        else:
            raise NotImplementedError(
            "Provide your own dataset or run with --use-dummy-data to test the pipeline."
            )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
