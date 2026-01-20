from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from src.data_loaders.data_util_functions import filter_and_normalize, crop_3d_spatial_bounding_box

class ISLESDataset(Dataset):
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
        self.include_first = True
        self.include_last = True
        # check if we want to load / save the data
        self.load_data = kwargs.get('load_data_isles', True)  # load the data again once the
        print(f"loading data: {self.load_data}, else: we generate it and save it")
        if self.load_data:
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
            self.precompute_randomness()
        print('data loaded')

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
        data = np.transpose(self.data[index], (3, 2, 1, 0))
        sampled_data = data[:, np.newaxis]
        sampled_data = filter_and_normalize(sampled_data)# data[::5, np.newaxis] was the previous one!
        t_full = sampled_data.shape[0]
        if self.precompute_random:
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
