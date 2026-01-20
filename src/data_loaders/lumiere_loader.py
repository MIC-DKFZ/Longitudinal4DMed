from torch.utils.data import Dataset
import os, sys
import numpy as np
import random
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from src.utils.plotting import *
from src.data_loaders.data_util_functions import filter_and_normalize, filter_and_normalize


class LumiereDataset(Dataset):
    def __init__(self, data_name='Lumiere',in_shape = None ,data_dir=None, train_test_val='trn', debug: bool = True, **kwargs):
        super(LumiereDataset, self).__init__()
        self.debug = debug
        self.noise = kwargs.get('noise', 0)
        self.num_to_keep = kwargs.get('num_to_keep_context', 1)
        self.train_test_val_mode = train_test_val
        self.data_name = data_name
        self.data_path = os.getenv('DATASET_LOCATION', 'data/') if data_dir is None else data_dir
        self.data_raw = {}
        self.data = {}
        self.mask_training = kwargs.get('mask_training', False)
        self.max_time = 255
        self.hparams = kwargs
        self.min_context = kwargs.get('data_time_embed_resolution', 16)
        self.two_dim = False
        self.data_save = {}
        self.counter_patient = 0
        # check if we want to load / save the data
        self.load_data = True
        print(f"loading data {self.load_data}, if not we generate it")
        self.in_shape = kwargs.get('in_shape', (8, 1, 96, 96, 64)) if in_shape is None else in_shape
        self.img_shape = self.in_shape[2:]  # D, H, W
        if self.load_data:
            self.load_data_from_file_single_npy()
        else:
            self.generate_save_data()
            # a little bit of a hack here
            np.save(os.path.join(self.data_path, f'trn_lumiere.npy'),
                    [self.data[key] for key in list(self.data.keys())[:50]], allow_pickle=True)
            np.save(os.path.join(self.data_path, f'val_lumiere.npy'),
                    [self.data[key] for key in list(self.data.keys())[50:60]], allow_pickle=True)
            np.save(os.path.join(self.data_path, f'tst_lumiere.npy'),
                    [self.data[key] for key in list(self.data.keys())[60:]], allow_pickle=True)
        print('data loaded')

    def __len__(self):
        return len(self.data)

    def load_data_from_file_single_npy(self):
        # here we only load the data we actually need
        trn_val = self.train_test_val_mode
        val_split = self.hparams.get('val_split', 0)
        if trn_val == 'tst':
            loc_data_unclean = np.load(os.path.join(self.data_path, f'{trn_val}_lumiere.npy'), allow_pickle=True)[()]
        else:
            loc_data_unclean_trn = np.load(os.path.join(self.data_path, f'trn_lumiere.npy'), allow_pickle=True)[()]
            loc_data_unclean_val = np.load(os.path.join(self.data_path, f'val_lumiere.npy'), allow_pickle=True)[()]
            loc_data_unclean = np.concatenate((loc_data_unclean_trn, loc_data_unclean_val), axis=0)
        # due to some errors there are images which are empty, we need to filter that!
        cleaned_data = []
        # we have to clean the data, in order for it
        for patient in loc_data_unclean:
            is_valid = False
            if len(patient) == 1:
                continue
            for visit in patient.values():
                if len(visit) != 3:
                    break
                img, seg, time = visit
                if not isinstance(img, np.ndarray) or img.size == 0:
                    break
                if not isinstance(seg, np.ndarray) or seg.size == 0:
                    break
                if isinstance(time, str):
                    break
                if np.any(img > 0.05):  # replicate [:2] > 0.05 check
                    is_valid = True
            if is_valid:
                cleaned_data.append(patient)

        if trn_val != "tst":
            if trn_val == "trn":
                cleaned_data = [
                    case for i, case in enumerate(cleaned_data)
                    if i % 5 != val_split
                ]
            else:  # mode == "val"
                cleaned_data = [
                    case for i, case in enumerate(cleaned_data)
                    if i % 5 == val_split
                ]

        self.data = cleaned_data
        return

    def __getitem__(self, index):
        data = self.data[index]  # All visits for this sample
        times, imgs, segs = [], [], []

        # Extract times, images, and segmentations
        for visit in data:
            img, seg, time_week = data[visit]  # FIX: Use direct indexing
            img = filter_and_normalize(img)
            times.append(time_week)
            imgs.append(img)
            segs.append(seg)

        # Sort by time
        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices] / self.max_time
        imgs = np.array(imgs, dtype=np.float32)[sorted_indices]
        segs = np.array(segs, dtype=np.float32)[sorted_indices]

        # Identify the target (latest time point)
        target = imgs[-1]
        target_seg = segs[-1]
        target_time = times[[-1]]
        # Context time and images, just the ones before the target
        # randomly mask them as well?
        context = imgs[:-1]
        context_seg = segs[:-1]
        context_time = times[:-1]
        # since we want to possibly mask the training
        if self.train_test_val_mode == 'trn' and self.mask_training:
            num_context = context.shape[0]
            if self.num_to_keep >= num_context:
                indices_to_keep = list(range(num_context))
            else:
                candidate_indices = list(range(num_context - self.num_to_keep))
                if len(candidate_indices) > 1:
                    num_to_keep_random = random.randint(1, len(candidate_indices) - 1)
                else:
                    num_to_keep_random = 0
                random.shuffle(candidate_indices)
                indices_to_keep = candidate_indices[:num_to_keep_random]

            # Instead of zeroing out non-selected context frames, slice to only keep the selected ones.
            context = context[indices_to_keep]
            context_seg = context_seg[indices_to_keep]
            context_time = context_time[indices_to_keep]

        # check if number of context is above 8, then we need to sample / pad
        num_context = len(context_time)
        if num_context > self.min_context:
            # Truncate to the last min_context points
            context = context[-self.min_context:]
            context_seg = context_seg[-self.min_context:]
            context_time = context_time[-self.min_context:]
        elif len(context_time) == 0 or len(context) == 0:
            # print('No context found, filling with ones')
            target = np.ones((1, 1, *self.img_shape), dtype=np.float32)
            target_seg = np.ones((1, 1, *self.img_shape), dtype=np.float32)
            context = np.ones((self.min_context, 1, *self.img_shape), dtype=np.float32)
            context_seg = np.ones((self.min_context, 1, *self.img_shape), dtype=np.float32)
            #target_time = 1
            target_time = np.float32(target_time)
            context_time = np.ones(self.min_context, dtype=np.float32)
            return (target, context, target_seg, context_seg, target_time, context_time)
        elif num_context < self.min_context:
            # Pad with zeros
            pad_size = self.min_context - num_context
            context = np.pad(context, ((pad_size, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            context_seg = np.pad(context_seg, ((pad_size, 0), (0, 0), (0, 0), (0, 0)), mode='constant',
                                 constant_values=0)
            context_time = np.pad(context_time, (pad_size, 0), mode='constant', constant_values=0)
        # make sure all is in float 32
        target = np.float32(target)
        target_time = np.float32(target_time)
        context = np.float32(context)
        context_time = np.float32(context_time)
        return {"target_img": target[np.newaxis, np.newaxis, ...], "context": context[:, np.newaxis, ...],
                "target_seg": target_seg[np.newaxis, np.newaxis, ...], "context_seg": context_seg[:, np.newaxis, ...],
                "target_time": np.array(target_time, dtype=np.float32),
                "context_time": np.array(context_time, dtype=np.float32)}

    def generate_save_data(self):
        '''
        load the lumiere data, for this, only single context single target?
        '''
        # get maximal number of patients
        patient_counter = 0
        # if self.train_test_val_mode =='trn':
        data_dir = 'data/lumiere/Lumiere'
        registered_images_dir = self.data_path + 'lumiere/Lumiere/images_registered/'
        import json
        # import the patient directory and the pre-processing
        with open(data_dir + '/patients.json') as f:
            patient_dict = json.load(f)
        ratings = pd.read_csv(data_dir + '/ratings.csv')
        # whyyy is this title so long?
        cases_total = {}
        ratings.rename(
            columns={'Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response,'
                     ' CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)': "rano_rating"},
            inplace=True)
        for patient in patient_dict.keys():
            try:
                acquisition_counter = 0
                result_patient_dict = {}
                patient_time_list = [patient_and_time for patient_and_time in
                                     os.listdir(data_dir + '/images_registered/') if
                                     patient.title().replace('_', '-') in patient_and_time]
                if self.debug and patient_counter > 4:
                    print(patient_counter, self.debug)
                    break
                for patient_time in patient_time_list:
                    # read the time
                    time = int(patient_time.split('-')[-1].split('_')[0])
                    image = nib.load(
                        data_dir + '/images_registered/' + patient_time + '/' + patient_time + '_0000.nii.gz').get_fdata()
                    segmentation = nib.load(
                        data_dir + '/segmentations_registered/' + patient_time + '.nii.gz').get_fdata()
                    resize_shape = (96, 96, 64)
                    image_resized = zoom(filter_and_normalize(image), zoom=np.divide(resize_shape, image.shape))
                    segmentation_resized = zoom(segmentation, zoom=np.divide(resize_shape, segmentation.shape))
                    result_patient_dict[acquisition_counter] = image_resized, segmentation_resized, time
                    acquisition_counter += 1
                cases_total[patient_counter] = result_patient_dict
                patient_counter += 1
                # the code crashes at around 40 patients!
                print(f'Generating data for patient {patient}')
            except:
                print(f'not a valid dir for Patient: {patient}')
        self.data = cases_total
        print('finished reshaping  data')

    def _get_data_shape(self):
        # we can make this more elegant, but this is fine I guess? maybe just move this to the train
        T_all, C, D, H, W = self.in_shape
        return (T_all, C, D, H, W)

    def load_single_data(self, index):
        return self.data[index]


if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR", "./data/")
    hparams = {
        "num_to_keep_context": 5,
        "debug": True,
        "val_split": 0,
    }
    dataset = LumiereDataset(train_test_val='trn', data_dir=data_dir, debug=True)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    plot_and_save_numpy_array(sample['context'][-1,0,...,32])
    print(f"Sample keys: {sample.keys()}")
    print(f"Target image shape: {sample['target_img'].shape}")
    print(f"Context shape: {sample['context'].shape}")
