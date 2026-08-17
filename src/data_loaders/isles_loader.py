from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from .data_util_functions import filter_and_normalize, crop_3d_spatial_bounding_box
import random

class ISLESDataset(Dataset):
    '''
    Note, this is the updated ISLES dataset class!
    '''
    def __init__(self, data_name='isles', data_dir=None, train_test_val='trn', debug: bool = True, **kwargs):
        super(ISLESDataset, self).__init__()
        # for debugging
        # print(f"Generating {data_size} {data_name} data samples")
        # print(growth_params, regular, sparse)
        self.hparams = kwargs
        self.debug = debug
        self.trn_val_split = 0  # between 0 and 4
        self.frames = kwargs.get('num_frames', 7)
        self.noise = kwargs.get('data_noise', 0.0)
        self.num_to_keep = kwargs.get('num_to_keep_context', 3)
        self.distance = kwargs.get('context_target_distance', 3)
        self.in_shape = kwargs.get('in_shape', (self.frames, 1, 192, 192, 16))
        self.train_test_val_mode = train_test_val
        self.data_name = data_name
        # the data path will be later in the config file
        self.data_path = os.getenv('DATA_DIR', 'data') if data_dir is None else data_dir
        self.data_raw = {}
        self.data = {}
        self.data_save = {}
        self.dense = kwargs.get('dense', False)
        self.include_first = True
        self.include_last = True
        self.paper_process = True
        if self.paper_process:
            # use the pre-processing of the official paper, which may not be ideal, we added a more complex one
            # note that the newer version needs different hyperparameters
            print(64*'#')
            print("Using the paper pre-processing, see dataloader for details")
            print("Only works for the pre-processing")
            print(64*'#')

        # Real, per-structure segmentation target: a CBF/CBV perfusion-abnormality ROI
        # mask (percentile-threshold over the ISLES-2024 challenge's own perfusion maps
        # — no extra tooling needed, unlike OASIS's FSL FAST/FIRST dependency), plus the
        # challenge's own expert lesion_mask/lvo_mask annotations, exposed as
        # target_seg_lesion/target_seg_lvo for eval.py. Requires data prepared via the
        # per-patient isles3/npy_data/<patient>.npz layout (see isles_prepare.py) — the
        # old single flat trn_m_isles.npy/tst_m_isles.npy layout has no per-patient
        # segmentation at all, so 'real' mode automatically falls back to the
        # all-ones placeholder when that newer per-patient data isn't found.
        # TODO: review comment
        self.seg_target_mode = kwargs.get('isles_seg_mode', 'real')
        assert self.seg_target_mode in ('real', 'placeholder'), \
            f"isles_seg_mode must be 'real' or 'placeholder', got {self.seg_target_mode!r}"
        self.perf_percentile = kwargs.get('perf_percentile', 98)
        self._npz_dir = os.path.join(self.data_path, 'isles3', 'npy_data')
        self._npz_mode = self.seg_target_mode == 'real' and os.path.isdir(self._npz_dir) and \
            any(f.endswith('.npz') for f in os.listdir(self._npz_dir))

        if self._npz_mode:
            print("[ISLESDataset] Using a real CBF/CBV perfusion-based segmentation target "
                  "(target_seg) plus expert lesion_mask/lvo_mask. Note: the published "
                  "TFM/CRONOS papers' ISLES numbers were produced with an all-ones placeholder "
                  "instead — pass isles_seg_mode='placeholder' to reproduce that exactly.")
        else:
            print(f"[ISLESDataset] No per-patient .npz segmentation data found at "
                  f"{self._npz_dir} — falling back to the all-ones placeholder target_seg "
                  "(this matches what the published TFM/CRONOS papers used for ISLES). Run "
                  "isles_prepare.py to enable the real CBF/CBV perfusion-based segmentation target.")

        # check if we want to load / save the data
        self.load_data = kwargs.get('load_data_isles', True)  # load the data again once the
        print(f"loading data: {self.load_data}, else: we generate it and save it")
        if self._npz_mode:
            self.load_data_from_npz_dir()
        elif self.load_data:
            self.load_data_from_file_single_npy()
        else:
            self.generate_save_data()
            self.data = self.data_raw
            # after you filled data_dict
            cases = np.array(list(self.data.values()), dtype=object)
            # split once
            tst = cases[:int(0.2 * len(cases))]
            trn_val = cases[int(0.2 * len(cases)):]
            np.save(f"{self.data_path}/tst_m_isles.npy", tst, allow_pickle=True)
            np.save(f"{self.data_path}/trn_m_isles.npy", trn_val, allow_pickle=True)

        self.start_idx = {}
        # only load them for val and test
        if debug:
            self.data = self.data[:6]
        if self.train_test_val_mode == 'trn':
            self.precompute_random = False
            self.index_dict = None
        else:
            self.precompute_random = True
            self.index_dict = {}
            if self._npz_mode:
                self.precompute_randomness_npz()
            else:
                self.precompute_randomness()
        print('data loaded')

    def crop_3d_spatial_bounding_box(self, img, threshold=0.02):
        """
        Crops a 4D image tensor with dimensions [W, H, D, T] to remove spatial areas
        where the values are below the given threshold. The time dimension remains unchanged.

        Parameters:
        img (numpy.ndarray): Input array with shape [W, H, D, T]
        threshold (float): Threshold to define the bounding box.

        Returns:
        numpy.ndarray: Cropped image with minimal width, height, depth, but with original time dimension.
        """
        # Validate input dimensions
        assert len(img.shape) == 4, "Input image must have 4 dimensions [W, H, D, T]"
        W, H, D, T = img.shape

        # Find the bounding box that includes all regions with values greater than the threshold
        mask = np.max(img, axis=3) > threshold  # Aggregate across the time axis

        # Get the slices for width, height, depth where the value exceeds the threshold
        w_indices = np.any(mask, axis=(1, 2))  # Collapse H and D to find relevant W
        h_indices = np.any(mask, axis=(0, 2))  # Collapse W and D to find relevant H
        d_indices = np.any(mask, axis=(0, 1))  # Collapse W and H to find relevant D

        # Find the bounds for width, height, and depth
        min_w, max_w = np.where(w_indices)[0][[0, -1]]
        min_h, max_h = np.where(h_indices)[0][[0, -1]]
        min_d, max_d = np.where(d_indices)[0][[0, -1]]

        # Crop the image across width, height, and depth, keeping time constant
        cropped_img = img[min_w:max_w + 1, min_h:max_h + 1, min_d:max_d + 1, :]

        return cropped_img

    def filter_and_normalize(self, data, lower_percentile=2, upper_percentile=98):
        """
        Filters outliers and normalizes the data.

        Args:
            data (np.array): Input data to be filtered and normalized.
            lower_percentile (float): Lower percentile for filtering.
            upper_percentile (float): Upper percentile for filtering.

        Returns:
            np.array: Filtered and normalized data.
        """
        # Compute the lower and upper bounds for filtering
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)

        # Clip the data to remove outliers
        data_clipped = np.clip(data, lower_bound, upper_bound)

        # Normalize the data to range [0, 1]
        data_normalized = (data_clipped - data_clipped.min()) / (data_clipped.max() - data_clipped.min())

        return data_normalized

    def load_data_from_file_single_npy(self):
        trn_val_tst = self.train_test_val_mode
        val_split = self.hparams.get('val_split', 0)
        if trn_val_tst == 'tst':
            self.data = np.load(os.path.join(self.data_path, 'tst_m_isles.npy'))
        else:
            self.data = np.load(os.path.join(self.data_path, 'trn_m_isles.npy'))
            indices = np.arange(self.data.shape[0])
            if trn_val_tst == 'trn':
                self.data = self.data[indices % 5 != val_split]
            elif trn_val_tst == 'val':
                self.data = self.data[indices % 5 == val_split]
        # self.data = np.array([self.filter_and_normalize(case) for case in self.data])
        return

    def load_data_from_npz_dir(self):
        """Per-patient .npz layout (see isles_prepare.py): each file holds
        ctp/cbf/cbv/lvo_mask/lesion_mask/cow_mask for one patient, loaded lazily
        in __getitem__ rather than concatenated into one array up front."""
        all_patients = sorted([
            os.path.join(self._npz_dir, f) for f in os.listdir(self._npz_dir) if f.endswith('.npz')
        ])
        indices = np.arange(len(all_patients))
        val_split = self.hparams.get('val_split', 0)
        trn_val_tst = self.train_test_val_mode
        if trn_val_tst == 'tst':
            selected = [all_patients[i] for i in indices if i % 5 == (val_split + 1) % 5]
        elif trn_val_tst == 'trn':
            selected = [all_patients[i] for i in indices if i % 5 != val_split and i % 5 != (val_split + 1) % 5]
        else:  # 'val'
            selected = [all_patients[i] for i in indices if i % 5 == val_split]
        self.data = selected
        print(f'  {trn_val_tst}: {len(self.data)} patients (npz mode)')

    def _make_perfusion_mask(self, cbf, cbv):
        """Binary ROI mask: voxels above perf_percentile in CBF OR CBV (positive voxels only)."""
        p = self.perf_percentile
        cbf_pos = cbf[cbf > 0]
        cbv_pos = cbv[cbv > 0]
        cbf_thresh = cbf > np.percentile(cbf_pos, p) if cbf_pos.size > 0 else np.zeros_like(cbf, dtype=bool)
        cbv_thresh = cbv > np.percentile(cbv_pos, p) if cbv_pos.size > 0 else np.zeros_like(cbv, dtype=bool)
        return (cbf_thresh | cbv_thresh).astype(np.float32)

    @staticmethod
    def _pad_spatial(arr, target_hw):
        """Zero-pad H and W (first two axes) to target_hw. Supports (H,W,D) and (H,W,D,T)."""
        target_h, target_w = target_hw
        H, W = arr.shape[:2]
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)
        pads = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
        pads += [(0, 0)] * (arr.ndim - 2)
        return np.pad(arr, pads)

    _MIN_GAP = 3  # minimum frames between last context frame and target frame
    _MAX_GAP = 8  # maximum gap (clipped to sequence length)

    def _sample_frame_indices_npz(self, t_full):
        """Sample self.frames random context positions and a target at least _MIN_GAP later."""
        max_ctx = t_full - self._MIN_GAP - 1
        max_ctx = max(max_ctx, self.frames)
        pool = min(max_ctx, t_full - self._MIN_GAP)
        ctx_indices = sorted(np.random.choice(pool, size=self.frames, replace=False).tolist())
        last_ctx = ctx_indices[-1]
        min_tgt = last_ctx + self._MIN_GAP
        target_frame_idx = min_tgt
        return ctx_indices, target_frame_idx

    def _create_missing_mask_npz(self):
        """Binary mask of shape (self.frames,) indicating which context slots are visible."""
        last_context = self.frames - self.distance
        missing_mask = np.zeros(self.frames, dtype=np.float32)
        n_keep = min(max(2, self.num_to_keep), last_context)
        keep_idx = np.random.choice(last_context, size=n_keep, replace=False)
        missing_mask[keep_idx] = 1
        if self.include_first:
            missing_mask[0] = 1
        if self.include_last:
            missing_mask[last_context] = 1
        return missing_mask

    def precompute_randomness_npz(self):
        """Precompute random frame indices and missing mask for deterministic val/test."""
        print("Precomputing random indices for deterministic validation and test (npz mode)")
        self.indices_random = []
        for patient_path in self.data:
            npz = np.load(patient_path)
            t_orig = npz['ctp'].shape[-1]
            t_full = (t_orig + 1) // 2  # matches [::2] subsampling in __getitem__
            ctx_indices, target_frame_idx = self._sample_frame_indices_npz(t_full)
            missing_mask = self._create_missing_mask_npz()
            self.indices_random.append({
                'ctx_indices': ctx_indices,
                'target_frame_idx': target_frame_idx,
                'missing_mask': missing_mask,
            })

    def _getitem_npz(self, index):
        npz = np.load(self.data[index])
        ctp = npz['ctp'].astype(np.float32)[..., ::2]  # (H, W, D, T), cap T
        lvo_mask = npz['lvo_mask'].astype(np.float32)
        lesion_mask = npz['lesion_mask'].astype(np.float32)
        # Perfusion mask computed on the fly so the threshold can be tuned without re-running preprocessing.
        # TODO: review comment
        if 'cbf' in npz and 'cbv' in npz:
            perfusion_mask = self._make_perfusion_mask(npz['cbf'].astype(np.float32), npz['cbv'].astype(np.float32))
        else:
            perfusion_mask = np.ones(ctp.shape[:3], dtype=np.float32)

        target_hw = self.in_shape[2:4]  # (H, W)
        ctp = self._pad_spatial(ctp, target_hw)
        lvo_mask = self._pad_spatial(lvo_mask, target_hw)
        lesion_mask = self._pad_spatial(lesion_mask, target_hw)
        perfusion_mask = self._pad_spatial(perfusion_mask, target_hw)

        data = np.transpose(ctp, (3, 2, 1, 0))  # (T, D, W, H)
        sampled_data = data[:, np.newaxis]  # (T, 1, D, W, H)
        t_full = sampled_data.shape[0]

        # ctp is reversed (H,W,D,T)->(T,D,W,H) above, so the masks (still H,W,D) must
        # reverse the same way (2,1,0)->(D,W,H) to stay aligned — otherwise H/W end up
        # swapped, invisible on square (H==W) volumes (same bug class as ACDC's seg fix).
        # TODO: review comment
        lvo_mask = np.transpose(lvo_mask, (2, 1, 0))
        lesion_mask = np.transpose(lesion_mask, (2, 1, 0))
        perfusion_mask = np.transpose(perfusion_mask, (2, 1, 0))

        if self.precompute_random:
            rand_info = self.indices_random[index]
            ctx_indices = rand_info['ctx_indices']
            target_frame_idx = rand_info['target_frame_idx']
            missing_mask = rand_info['missing_mask']
        else:
            ctx_indices, target_frame_idx = self._sample_frame_indices_npz(t_full)
            missing_mask = self._create_missing_mask_npz()

        context = sampled_data[ctx_indices]        # (T, 1, D, W, H)
        target = sampled_data[[target_frame_idx]]  # (1, 1, D, W, H)

        context_time = np.array(ctx_indices, dtype=np.float32) / max(t_full - 1, 1)
        target_time = np.array([target_frame_idx], dtype=np.float32) / max(t_full - 1, 1)

        context = context * missing_mask[..., None, None, None, None]
        noise = np.random.randn(*context.shape) * self.noise
        context = np.float32(context + noise)

        # target_seg/context_seg/target_seg_* match target_img's own shape (1,1,D,W,H)
        # exactly, same convention this file's old placeholder used (np.ones(shape=
        # target.shape)) and the same one oasis_loader.py uses — not a "missing
        # channel dim" convention, so no unsqueeze is needed downstream.
        # TODO: review comment
        target_seg = perfusion_mask[np.newaxis, np.newaxis, ...].astype(np.float32)
        target_seg_lesion = lesion_mask[np.newaxis, np.newaxis, ...].astype(np.float32)
        target_seg_lvo = lvo_mask[np.newaxis, np.newaxis, ...].astype(np.float32)

        return {
            'target_img': target,
            'context': context,
            'target_seg': target_seg,
            'context_seg': target_seg,
            # Per-structure masks so eval.py can optionally report per-structure seg-masked metrics.
            # TODO: review comment
            'target_seg_lesion': target_seg_lesion,
            'target_seg_lvo': target_seg_lvo,
            'target_time': target_time,
            'context_time': context_time,
        }

    def __len__(self):
        return len(self.data)



    def precompute_randomness(self):
        """Precompute the random target index and missing mask for each sample.
        See paper for details
        """
        print("Precomputing random target index and missing mask for deterministic validation and test")
        self.indices_random = []
        N = len(self.data)
        for idx in range(N):
            missing_mask, target_idx = self._create_missing_mask()
            self.indices_random.append({
                'target_idx': target_idx,
                'missing_mask': missing_mask
            })

    def _create_missing_mask(self):
        """
        a function to make sure that get item has a consistent masking strategy
        :return: a missing mask and the target index
        """
        target_idx = np.random.randint(self.frames, self.frames * 3)
        # Create the missing_mask with random entries (length = target_idx - distance)
        last_context = self.frames - self.distance
        missing_mask = np.zeros(self.frames, dtype=np.float32)
        missing_bits = np.random.randint(0, 2, size=last_context).tolist()
        missing_mask[:last_context] = missing_bits

        if self.include_first:
            missing_mask[0] = 1
        if self.include_last:
            missing_mask[last_context] = 1
        # safety
        if not any(missing_mask):
            missing_mask[np.random.randint(last_context)] = 1

        return missing_mask, target_idx

    def __getitem__(self, index):
        if self._npz_mode:
            return self._getitem_npz(index)
        data = np.transpose(self.data[index], (3, 2, 1, 0))
        sampled_data = data[:, np.newaxis]
        sampled_data = filter_and_normalize(sampled_data)# data[::5, np.newaxis] was the previous one!
        t_full = sampled_data.shape[0]
        if self.dense:
            # Return the full sequence: all frames as context, last frame as target
            context = sampled_data[:-1]
            target = sampled_data[[-1]]
            time_full = np.linspace(0, 1, t_full, dtype=np.float32)
            return {'target_img': target, 'context': context,
                    'target_seg': np.ones(shape=target.shape),
                    'context_seg': np.ones(shape=target.shape),
                    'target_time': time_full[[-1]],
                    'context_time': time_full[:-1]}
        elif self.precompute_random:
            rand_info = self.indices_random[index]
            target_idx = rand_info['target_idx']
            missing_mask = rand_info['missing_mask']
        else:
            missing_mask, target_idx = self._create_missing_mask()
        time_full = np.linspace(0, 1, t_full, dtype=np.float32)
        start_index = int(target_idx - self.frames)
        end_index = int(target_idx)
        context = sampled_data[start_index:end_index]
        target = sampled_data[[end_index]]

        context = context * missing_mask[..., None, None, None, None]
        context_time = time_full[start_index:end_index]
        target_time = time_full[[end_index]]
        # Add small noise to the context if
        noise = np.random.randn(*context.shape) * self.noise
        context = context + noise
        context = np.float32(context)
        return {'target_img': target, 'context': context, 'target_seg': np.ones(shape=target.shape),
                'context_seg': np.ones(shape=target.shape),
                "target_time": target_time, "context_time": context_time}

    def generate_save_data(self):
        '''
        Generate and save data: we load Brats data, pick a slice (for now), and save it
        '''
        # get maximal number of patients
        patient_counter = 0
        data_dir_a = 'data/isles/isles24_train_a/raw_data'
        data_dir_b = 'data/isles/isles24_train_b/raw_data'

        # only do this if you have the local memory
        data_dirs = [data_dir_a, data_dir_b]
        for data_dir in data_dirs:
            for patient in os.listdir(data_dir):
                # have the debug option
                #if self.debug and patient_counter > 4:
                #    print(patient_counter, self.debug)
                #    break
                print(f'Generating data for patient {patient}')
                # load the p CT  data here

                if self.paper_process:
                    patient_dir = os.path.join(data_dir, patient, 'ses-01')
                    patient_images_ct = os.listdir(patient_dir)
                    patient_image_ctp_str = [modality for modality in patient_images_ct if 'ctp' in modality][0]
                    img = nib.load(os.path.join(patient_dir, patient_image_ctp_str)).get_fdata().astype(np.float32)
                    img = self.filter_and_normalize(img)
                    img[img > 0.80] = 0
                    # crop around the non-important slices
                    img = self.crop_3d_spatial_bounding_box(img)
                    resize_shape = (128, 128, 16, 32)
                    img_shaped = zoom(img, zoom=np.divide(resize_shape, img.shape))
                else:
                    patient_dir = os.path.join(data_dir, patient, 'ses-01')
                    patient_images_ct = os.listdir(patient_dir)
                    patient_image_ctp_str = [modality for modality in patient_images_ct if 'ctp' in modality][0]
                    perfusion_map_location = os.path.join(patient_dir, 'perfusion-maps')
                    # we load tmax from this dir
                    patient_image_tmax_str = \
                    [modality for modality in os.listdir(perfusion_map_location) if 'tmax' in modality][0]
                    img = nib.load(os.path.join(patient_dir, patient_image_ctp_str)).get_fdata().astype(np.float32)
                    tmax_image = nib.load(os.path.join(perfusion_map_location, patient_image_tmax_str)).get_fdata().astype(
                        np.float32)
                    # tmax as a quasi mask
                    img_masked = img * (tmax_image > tmax_image.min())[..., None]
                    img = filter_and_normalize(img_masked)
                    # crop around the non-important slices
                    img = crop_3d_spatial_bounding_box(img)
                    resize_shape = (192, 192, 16, 32)  # todo: into args
                    img_shaped = zoom(img, zoom=np.divide(resize_shape, img.shape))
                    self.data_raw[patient_counter] = img_shaped
                    patient_counter += 1
                    print(img.shape)
                print('finished patient number: ', patient_counter)
                # Import images from directory
            print('finished generating data')
    def load_single_data(self, index):
        return self.data[index]

    def _get_data_shape(self):
        '''
        Used to build the model with the correct input shape
        :return:
        '''
        if self._npz_mode:
            sample = self[0]
            _, C, D, H, W = sample['context'].shape
            return (self.frames, C, D, H, W)
        # we can make this more elegant, but this is fine I guess? maybe just move this to the train
        # T_all, C, D, H, W = self.in_shape
        data_sample = np.transpose(self.data[0], (3, 2, 1, 0))
        T,D,H,W = data_sample.shape
        return (self.frames, 1, D, H, W)

    #    def _get_data_shape(self):
        # we can make this more elegant, but this is fine I guess? maybe just move this to the train

        # T_all, C, D, H, W = self.data.shape[1:]
        # return (int(T_all-1), C, D, H, W)


if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR", "./data/")
    hparams = {
        "num_to_keep_context": 5,
        "debug": True,
        "val_split": 0,
    }
    dataset = ISLESDataset(data_dir=data_dir, train_test_val="val", **hparams)
    print(f"Dataset length: {len(dataset)}")
    example = next(iter(dataset))
    print("Sampled one data point successfully.")
