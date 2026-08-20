import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.draw import polygon
from torch.utils.data import Dataset

from .data_util_functions import filter_and_normalize

_SPLIT_MAP = {'trn': 'TRAIN', 'val': 'VAL', 'tst': 'TEST'}


class AimiDataset(Dataset):
    """EchoNet-Dynamic-based echocardiography dataset.
    
    NOTE: this dataset is 2D, but a lot of data. 
    Not really tested, but may be useful to some. 

    VolumeTracings.csv gives an expert LV boundary trace at exactly two frames
    per video (ED and ES). Frame sampling is built around those two frames 
    specifically. Similar usage as acdc. 
    """

    def __init__(self, data_dir=None, train_test_val='trn', debug: bool = False, resize=None, **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.debug = debug
        self.train_test_val_mode = train_test_val
        self.data_path = Path(os.getenv('DATASET_LOCATION_AIMI', 'data') if data_dir is None else data_dir)
        self.videos_dir = self.data_path / 'Videos'
        self.resize = resize
        self.noise = kwargs.get('data_noise', 0.0)
        self.num_context = kwargs.get('num_to_keep_context', 5)

        file_list = pd.read_csv(self.data_path / 'FileList.csv')
        split = _SPLIT_MAP[train_test_val]
        file_list = file_list[file_list['Split'] == split]

        tracings = pd.read_csv(self.data_path / 'VolumeTracings.csv')
        # VolumeTracings.csv's FileName includes '.avi'. 
        #FileList.csv's doesn't  strip it here so both dicts key off the same (suffix-less) video identifier.
        tracings = tracings.assign(FileName=tracings['FileName'].str.replace('.avi', '', regex=False))
        # Two traced frames per video (ED/ES): {FileName: {frame: (N,4) x1,y1,x2,y2 chords}}.
        self.traces = {
            fname: {
                frame: grp[['X1', 'Y1', 'X2', 'Y2']].to_numpy(dtype=np.float32)
                for frame, grp in sub.groupby('Frame')
            }
            for fname, sub in tracings.groupby('FileName')
        }

        self.files = [
            fname for fname in file_list['FileName']
            if (self.videos_dir / f"{fname}.avi").is_file()
            and fname in self.traces and len(self.traces[fname]) == 2
        ]
        if not self.files:
            raise RuntimeError(f"No usable videos found in {self.videos_dir}")
        if self.debug:
            self.files = self.files[:16]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _trace_to_mask(points, height, width):
        """points: (N, 4) array of (x1,y1,x2,y2) chords, base-to-apex. Standard
        EchoNet-Dynamic rasterization: walk down one side of the chords then
        back up the other to form a closed LV boundary polygon.
        """
        x1, y1, x2, y2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        xs = np.concatenate([x1, x2[::-1]])
        ys = np.concatenate([y1, y2[::-1]])
        rr, cc = polygon(ys, xs, shape=(height, width))
        mask = np.zeros((height, width), dtype=np.float32)
        mask[rr, cc] = 1.0
        return mask

    def __getitem__(self, index):
        fname = self.files[index]
        frames_traced = sorted(self.traces[fname].keys())
        frame_ctx, frame_tgt = frames_traced[0], frames_traced[1]  # earlier -> later, chronological

        cap = cv2.VideoCapture(str(self.videos_dir / f"{fname}.avi"))
        if not cap.isOpened():
            raise IOError(f"Could not open {fname}.avi")

        # Context: num_context frames sampled uniformly between frame 0 and ctx, so last frame is frame_ctx.
        ctx_indices = np.linspace(0, frame_ctx, self.num_context).round().astype(int)
        needed = sorted(set(ctx_indices.tolist() + [frame_tgt]))

        frames = {}
        idx = 0
        while len(frames) < len(needed):
            ok, f = cap.read()
            if not ok:
                break
            if idx in needed:
                if f.ndim == 3 and f.shape[-1] == 3:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if self.resize is not None:
                    f = cv2.resize(f, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
                frames[idx] = f.astype(np.float32) / 255.0
            idx += 1
        cap.release()

        height, width = frames[frame_tgt].shape
        context = np.stack([frames[i] for i in ctx_indices], axis=0)  # (T-1, H, W)
        target = frames[frame_tgt][None]  # (1, H, W)

        context = filter_and_normalize(context)
        target = filter_and_normalize(target)
        noise = np.random.randn(*context.shape).astype(np.float32) * self.noise
        context = context + noise

        target_seg = self._trace_to_mask(self.traces[fname][frame_tgt], height, width)
        ctx_seg_last = self._trace_to_mask(self.traces[fname][frame_ctx], height, width)
        context_seg = np.zeros_like(context)
        context_seg[-1] = ctx_seg_last  # only the last context frame (frame_ctx) has a real trace

        # add channel + depth (D=1) dims: (T, H, W) -> (T, 1, 1, H, W)
        context = context[:, None, None, ...]
        context_seg = context_seg[:, None, None, ...]
        target = target[:, None, None, ...]
        target_seg = target_seg[None, None, None, ...]

        num_frames = max(frame_tgt, 1)
        context_time = (ctx_indices / num_frames).astype(np.float32)
        target_time = np.array([frame_tgt / num_frames], dtype=np.float32)

        return {
            "target_img": torch.from_numpy(target.astype(np.float32)),
            "context": torch.from_numpy(context.astype(np.float32)),
            "target_seg": torch.from_numpy(target_seg.astype(np.float32)),
            "context_seg": torch.from_numpy(context_seg.astype(np.float32)),
            "target_time": target_time,
            "context_time": context_time,
        }

    def _get_data_shape(self):
        sample = self[0]
        T, C, D, H, W = sample['context'].shape
        return (T, C, D, H, W)


if __name__ == "__main__":
    ds = AimiDataset(train_test_val='val', debug=True)
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print({k: getattr(v, 'shape', v) for k, v in sample.items()})
