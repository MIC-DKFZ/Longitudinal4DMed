from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from src.data_loaders.data_util_functions import crop_3d_spatial_bounding_box, filter_and_normalize


class OASISDataset(Dataset):
    def __init__(self, train_test_val='trn',data_dir=None, debug: bool = True, **kwargs):
        super().__init__()
        self.debug = debug
        self.noise = kwargs.get('noise', 0)
        self.num_to_keep = kwargs.get('num_to_keep_context', 1)
        self.train_test_val_mode = train_test_val
        self.data_path = os.get('DATASET_LOCATION_ISLES', 'data') if data_dir is None else data_dir
        self.max_time = kwargs.get('max_time', 1.0)  # here: just normalize to [0,1] using visit index
        self.frames = kwargs.get('frames', 16)
        self.hparams = kwargs
        self.min_context = kwargs.get('data_time_embed_resolution', 16)
        self.in_shape = kwargs.get('in_shape', (5, 1, 128, 128, 160))
        #self.img_shape = kwargs.get('in_shape', (5, 1, 96, 96, 96))[2:]
        #self.num_times = kwargs.get('in_shape', (5, 1, 96, 96, 96))[0]
        self.data = []
        self.max_time = 0
        self.build_oasis_cases()
        print(f"OASIS: built {len(self.data)} longitudinal samples")

    def __len__(self):
        return len(self.data)

    def filter_and_normalize(self, data, lower_percentile=2, upper_percentile=98):
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        data_clipped = np.clip(data, lower_bound, upper_bound)
        denom = (data_clipped.max() - data_clipped.min())
        if denom == 0:
            return np.zeros_like(data_clipped, dtype=np.float32)
        data_normalized = (data_clipped - data_clipped.min()) / denom
        return data_normalized.astype(np.float32)

    def resize_3d(self, img, target_shape, order=1):
        # img: (D, H, W) or (H, W, D)
        # target_shape: (D_new, H_new, W_new)
        zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
        return zoom(img, zoom_factors, order=order).astype(np.float32)

    def resize_4d(self,img_4d, target_shape, order=1):
        # img_4d: (H, W, D, T) or (D, H, W, T)
        out = []
        for t in range(img_4d.shape[-1]):
            out.append(self.resize_3d(img_4d[..., t], target_shape, order=order))
        return np.stack(out, axis=-1).astype(np.float32)

    def build_oasis_cases(self):
        local_path = os.path.join(self.data_path, "oasis/processed")
        # read the metadata xlsx
        meatadata_frame = pd.read_excel(os.path.join(self.data_path,
                                                     "oasis/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"))
        # metadata_file = np.load
        subjects = np.array(sorted(os.listdir(local_path)))  # make it an array

        val_split = self.hparams.get("val_split", 0)
        mode = self.train_test_val_mode
        idx = np.arange(len(subjects))

        if mode == "tst":
            subjects = subjects[idx % 5 == val_split]
        elif mode == "trn":
            subjects = subjects[idx % 5 != val_split]
        else:  # "val"
            subjects = subjects[idx % 5 == val_split]

        subjects = subjects.tolist()  # back to list if you want

        if self.debug:
            subjects = subjects[:10]
        # max_day = 0
        for subj_idx, subj in enumerate(subjects):
            subj_dir = os.path.join(local_path, subj)
            visit_dirs = sorted(os.listdir(subj_dir))
            if len(visit_dirs) < 2:
                continue  # need at least 2 timepoints

            patient_dict = {}
            patient_arr = np.array([])
            date_arr = []
            age_arr = []
            for v_idx, visit in enumerate(visit_dirs):
                visit_dir = os.path.join(subj_dir, visit)
                if not os.path.isdir(visit_dir):
                    continue

                nii_files_name = visit_dir + '/t1_brain_MNI.nii.gz'
                if not os.path.isfile(nii_files_name):
                    continue

                img_nii = nib.load(nii_files_name)
                img = img_nii.get_fdata().astype(np.float32)

                # reshape to (D, H, W) if needed and resize
                if img.ndim == 4:
                    # pick first channel / volume
                    img = img[..., 0]
                # order to (H, W, D) -> (D, H, W)
                visit_num = int(visit.split('_')[-1])
                mri_id = 'OAS2_' + subj.split('_')[1] + '_MR' + str(visit_num)
                visit_date = meatadata_frame[meatadata_frame['MRI ID'] == mri_id]['MR Delay'].values
                age_visit = meatadata_frame[meatadata_frame['MRI ID'] == mri_id]['Age'].values
                if visit_date[0] > self.max_time:
                    self.max_time = visit_date
                date_arr.append(int(visit_date[0]))
                age_arr.append(age_visit)
                img -= np.min(img)
                img /= np.max(img)  # simple normalization
                # check if array is empty
                if patient_arr.size == 0:
                    patient_arr = img[...,None]
                else:
                    patient_arr = np.concatenate((patient_arr, img[...,None]), axis=-1)
            target_time_length, C, tgt_D, tgt_H, tgt_W = self.in_shape
            # potentially crop:
            imgs = crop_3d_spatial_bounding_box(patient_arr, threshold=0.01)
            # todo: the option for a random crop, or for reshaping to the desired size
            zooming = True
            if zooming:
                imgs= self.resize_4d(imgs, (tgt_D, tgt_H, tgt_W), order=1)
            else:
                imgs = self.random_crop(imgs, tgt_D, tgt_H, tgt_W)
            patient_dict['patient_arr'] = imgs
            patient_dict['subj_id'] = subj
            patient_dict['dates'] = date_arr
            patient_dict['ages'] = age_arr

            # patient_dict['times'] =

            if patient_arr.shape[-1] >= 2:
                self.data.append(patient_dict)

    def random_crop(self,img, tgt_D, tgt_H, tgt_W):
        # img either (D,H,W) or (T,D,H,W)
        has_time = (img.ndim == 4)

        if has_time:
            D, H, W, T = img.shape
        else:
            D, H, W = img.shape

        # choose random starts only if image is larger
        d0 = np.random.randint(0, D - tgt_D + 1) if D > tgt_D else 0
        h0 = np.random.randint(0, H - tgt_H + 1) if H > tgt_H else 0
        w0 = np.random.randint(0, W - tgt_W + 1) if W > tgt_W else 0

        # crop
        if has_time:
            img = img[d0:d0 + tgt_D, h0:h0 + tgt_H, w0:w0 + tgt_W, :]
        else:
            img = img[d0:d0 + tgt_D, h0:h0 + tgt_H, w0:w0 + tgt_W]

        return img

    def __getitem__(self, index):
        sample = self.data[index]
        imgs_4d = sample['patient_arr']  # (H, W, D, T)
        dates = np.array(sample['dates'], dtype=np.float32)  # len T

        # sort by time
        order = np.argsort(dates)
        imgs_4d = imgs_4d[..., order]
        dates = dates[order]

        # (H, W, D, T) -> (T, D, H, W)
        imgs = np.transpose(imgs_4d, (3, 2, 0, 1)).astype(np.float32)
        T, D, H, W = imgs.shape

        # split context / target
        target_time_length, C, tgt_D, tgt_H, tgt_W = self.in_shape
        # potentially crop:
        imgs = self.random_crop(imgs, tgt_D, tgt_H, tgt_W)
        imgs = self.filter_and_normalize(imgs, lower_percentile=1, upper_percentile=99)
        target = imgs[-1]  # (D, H, W)
        context = imgs[:-1]  # (T-1, D, H, W)

        # times in [0, 1]
        times = dates / max(self.max_time, 1e-6)
        target_time = times[[-1]]
        context_time = times[:-1]

        # stack first, then pad spatially to self.img_shape (e.g. 192^3)
        pad_d = max(0, tgt_D - D)
        pad_h = max(0, tgt_H - H)
        pad_w = max(0, tgt_W - W)
        # do this later into a single l
        if pad_d or pad_h or pad_w:
            target = np.pad(target,
                            ((0, pad_d), (0, pad_h), (0, pad_w)),
                            mode="constant")
            if context.shape[0] > 0:
                context = np.pad(context,
                                 ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                                 mode="constant")

        # context = self.random_crop(context, tgt_D, tgt_H, tgt_W)
        # target = self.random_crop(target, tgt_D, tgt_H, tgt_W)

        num_context = context.shape[0]
        if num_context < target_time_length:
            pad_t = target_time_length - num_context
            context = np.pad(
                context,
                ((pad_t, 0), (0, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            context_time = np.pad(
                context_time,
                (pad_t, 0),
                mode="constant",
                constant_values=0,
            )
        elif num_context > target_time_length:
            # keep last T context frames if there are too many
            context = context[-target_time_length:]
            context_time = context_time[-target_time_length:]

        # fake segs = zeros
        target_seg = np.zeros_like(target, dtype=np.float32)
        context_seg = np.zeros_like(context, dtype=np.float32)

        # add channel dim
        target = target[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
        target_seg = target_seg[np.newaxis, np.newaxis, ...]
        context = context[:, np.newaxis, ...]  # (T-1, 1, D, H, W)
        context_seg = context_seg[:, np.newaxis, ...]

        return {
            "target_img": target.astype(np.float32),
            "context": context.astype(np.float32),
            "target_seg": target_seg.astype(np.float32),
            "context_seg": context_seg.astype(np.float32),
            "target_time": target_time.astype(np.float32),
            "context_time": context_time.astype(np.float32),
        }
    def _get_data_shape(self):
        sample = self.data[0]
        imgs_4d = sample['patient_arr']  # (H, W, D, T)
        D, H, W, _ = imgs_4d.shape
        T_context = self.in_shape[0]
        return (T_context, 1, D, H, W)


if __name__ == "__main__":
    ds = OASISDataset()
    # dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    next(iter(ds))
