from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import random

from batchgenerators.augmentations.spatial_transformations import rotate_coords_3d
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates

from .data_util_functions import crop_3d_spatial_bounding_box, filter_and_normalize


def BraTS20_groupMapping(id: int, group_name=None):
    """Mapping between absolute ID numbers and dataset group IDs.
    Absolute IDs: 1 - 660
    Relative IDs: Training (369) [1-369], Validation (125) [370-494], Testing (166) [495-660]

    Params:
    - ID (int): Absolute or group ID
    - group_name (str): Accepts 'Training', 'Validation' and 'Testing'. Defaults to None.

    Return:
        If no `group_name` is provided, the ID is handled as absolute and the corresponding
        relative ID and group_name are returned.
        If a `group_name` is given, the ID is handled as relative to that group and the
        absolute ID along with the group name are returned."""

    groups = ["Training", "Validation", "Testing"]
    grp_lengths = [369, 125, 166]
    acc_lens = list(accumulate(grp_lengths))

    if group_name is None:
        assert 0 < id <= acc_lens[-1], \
            f'ID {id} is outside of absolute ID range. Expected ID in range: 1 <= ID <= {acc_lens[-1]}.'
        group_index = bisect_left(acc_lens, id)
        group_name = groups[group_index]
        if group_index > 0: id -= acc_lens[group_index - 1]
    else:
        assert group_name in groups, f'Unexpected group name was given: {group_name}. Expected: {groups}'
        group_index = groups.index(group_name)
        assert id <= grp_lengths[group_index], \
            f'ID {id} is out of range for relative IDs of {group_name} group. Expected a value between 1 and {grp_lengths[group_index]}'
        if group_index > 0: id += acc_lens[group_index - 1]

    return id, group_name

class SemiSynthLongi(Dataset):
    def __init__(self, data_dir=None, data_name='brats_longi', train_test_val='trn', **kwargs):
        super(SemiSynthLongi, self).__init__()
        # for debugging
        # print(f"Generating {data_size} {data_name} data samples")
        # print(growth_params, regular, sparse)
        self.num_patients = 0
        self.noise = kwargs.get('data_noise', 0.0)
        self.num_to_keep = kwargs.get('num_to_keep_context', 3)
        self.train_test_val_mode = train_test_val
        self.data_name = data_name
        # the data path will be later in the config file
        self.data_raw_path = os.getenv(
            'RAW_DATA_BR', '/data')
        self.data_path = data_dir if data_dir is not None else os.environ['DATA_DIR']
        self.data = {}
        self.two_dim = False
        self.data_save = {}
        self.counter_patient = 0
        self.memory_efficient = True
        self.hparams = kwargs
        self.time_length = 8
        self.min_max_time = [-3, 3]
        if self.memory_efficient:
            self.num_patients = 16  # = 64 +16+16 ; 369 later
        # check if we want to load / save the data
        self.load_data = True
        print("loading data, if not we generate it: ", self.load_data)

        if self.load_data:
            self.load_data_from_file_single_npy()
        else:
            print('generate Data')
            self.generate_save_data()
            # self.augment_data()
            # self.load_data = True
            trn = np.float32(np.stack([self.data[key][:, [0]] for key in list(self.data.keys())[:192]]))
            np.save(os.path.join(self.data_path, 'trn_brats_semi.npy'), trn)
            val = np.float32(np.stack([self.data[key][:, [0]] for key in list(self.data.keys())[192:224]]))
            np.save(os.path.join(self.data_path, 'val_brats_semi.npy'), val)
            tst = np.float32(np.stack([self.data[key][:, [0]] for key in list(self.data.keys())[224:256]]))
            np.save(os.path.join(self.data_path, 'tst_brats_semi.npy'), tst)
        if self.train_test_val_mode == 'trn':
            self.precompute_random = False
            self.indices_random = None
        else:
            self.precompute_random = True
            self.precompute_randomness()

        print('data loaded')

    def __len__(self):
        return len(self.data)

    def load_data_from_file_single_npy(self):
        trn_val = self.train_test_val_mode  # + 'data_semi/trn_val.npy'
        val_split = self.hparams.get('val_split', 0)

        if trn_val == 'tst':
            # For the test set, load only the test data.
            self.data = np.load(os.path.join(self.data_path, "tst_brats_semi.npy"), allow_pickle=True)
        else:
            # Merge training and validation sets.
            trn_data = np.load(os.path.join(self.data_path, "trn_brats_semi.npy"), allow_pickle=True)
            val_data = np.load(os.path.join(self.data_path, "val_brats_semi.npy"), allow_pickle=True)
            self.data = np.concatenate((trn_data, val_data), axis=0)

        # Apply filtering and normalization.
        self.data = np.array([filter_and_normalize(case) for case in self.data])

        # For training/validation, perform the bootstrap split based on remainder.
        if trn_val != 'tst':
            if trn_val == 'trn':
                self.data = [case for i, case in enumerate(self.data) if i % 5 != val_split]
            elif trn_val == 'val':
                self.data = [case for i, case in enumerate(self.data) if i % 5 == val_split]
        return

    def precompute_randomness(self):
        """Precompute the random target index and missing mask for each sample."""
        precomputed_masks = []
        for idx in range(len(self.data)):
            # For each sample, extract target and context
            sample = self.data[idx]
            # Assuming the first frame is the target and the remainder are context images:
            _, context = sample[:1], sample[1:]
            num_context = context.shape[0]

            if self.num_to_keep >= num_context:
                mask = np.ones(num_context, dtype=np.float32)
            else:
                # Randomly decide how many context images (from candidates) to keep.
                num_to_keep_random = random.randint(1, num_context - self.num_to_keep - 1)
                # Only allow masking on the first part (exclude the last self.num_to_keep indices)
                candidate_indices = list(range(num_context))[:-self.num_to_keep]
                random.shuffle(candidate_indices)
                indices_to_keep = candidate_indices[:num_to_keep_random]
                mask = np.zeros(num_context, dtype=np.float32)
                for i in indices_to_keep:
                    mask[i] = 1.0
            precomputed_masks.append(mask)
        self.indices_random = precomputed_masks
        return

    def __getitem__(self, index):

        target, context = self.data[index][:1], self.data[index][1:]
        num_context = context.shape[0]

        if self.precompute_random:
            mask = self.indices_random[index]
        else:
            # Compute randomness on the fly.
            if self.num_to_keep >= num_context:
                mask = np.ones(num_context, dtype=np.float32)
            else:
                num_to_keep_random = random.randint(1, num_context - self.num_to_keep - 1)
                candidate_indices = list(range(num_context))[:-self.num_to_keep]
                random.shuffle(candidate_indices)
                indices_to_keep = candidate_indices[:num_to_keep_random]
                mask = np.zeros(num_context, dtype=np.float32)
                for i in indices_to_keep:
                    mask[i] = 1.0

        # Apply the mask: for each context image index not marked in the mask, set that image to zeros.
        for i in range(num_context):
            if mask[i] == 0:
                context[i] = np.zeros_like(context[i])

        # Add small noise to the context (normally distributed noise)
        # Standard deviation of the noise # 0.01
        noise = np.random.randn(*context.shape) * self.noise * mask[(...,) + (None,) * (context.ndim - 1)]

        context = context + noise
        context = np.float32(context)
        # if there is no time vector, make a simple init
        time_vector = np.linspace(0, 1, context.shape[0] + 1, dtype=np.float32)
        return {"target_img": target, "context": context,
                "target_seg": np.ones_like(target), "context_seg": np.ones_like(target),
                "target_time": time_vector[[-1]], "context_time": time_vector[:-1]}

    def generate_save_data(self):
        '''
        Generate and save data: we load Brats data, pick a slice (for now), and save it
        '''
        # get maximal number of patients
        for i in range(self.num_patients):
            print(f'Generating data for patient {i}')
            rel_id, grp = BraTS20_groupMapping(i + 1)
            # Import images from directory
            FROM_NETWORK = True
            STUDY_COUNT = rel_id
            data = self.data_raw_path
            studyPath = "/BraTS20_" + grp + "_"  # #+grp+"Data"

            for study in range(STUDY_COUNT, STUDY_COUNT + 1):
                if FROM_NETWORK:
                    print('loading patient')
                    imgs = [
                        nib.load(data + (studyPath + str(study).zfill(3)) * 2 + f"_{m}.nii.gz").get_fdata().astype(
                            np.float32)[
                        :, :, :] \
                        for m in ["t1", "seg"]]
                    if grp == 'Training':
                        l_head = nib.load(data + (studyPath + str(study).zfill(3)) * 2 + "_seg.nii.gz")
                        print(l_head.header.get_data_shape(), l_head.header.get_xyzt_units()[0],
                              nib.affines.voxel_sizes(l_head.affine).tolist(), list(nib.aff2axcodes(l_head.affine)))
                        label = nib.load(
                            data + (studyPath + str(study).zfill(3)) * 2 + "_seg.nii.gz").get_fdata().astype(
                            np.uint8)[:, :, :]  # H, W, D
                else:  # from local BraTS directory
                    imgs = [np.load(data + studyPath + str(study).zfill(3) + f"_{m}.npy") for m in
                            ["x", "y", "meta", "orig_lbl"]]

                #
                labels = np.zeros(label.shape)
                labels[label == 1] = 1  # Necrotic, non-enhancing core (NCR) (Center)
                labels[label == 2] = 1  # Peritumoral Edema (ED) (Large area beyond NCR and Enhancing Tumor)
                labels[label == 4] = 1  # Enhancing Tumor (Closest Area around NCR)
                self.apply_augs_and_save(imgs, i)

    def apply_augs_and_save(self, img_raw, pat_id):
        # the latent trajectory resolution
        time_points = np.linspace(self.min_max_time[0], self.min_max_time[1], self.time_length)
        # essentially only single channel
        if self.memory_efficient:
            # so that we only have the first and last time point
            # we also want them to be in the same order
            modality_indices = [0, len(img_raw) - 1]
        else:
            # get all the modalities
            modality_indices = range(len(img_raw))
        for j in range(14):
            # different augmentations
            augm_imgs = {modality: self.deform_structure(modality, img_raw[modality], label_img=img_raw[-1],
                                                         intensity=10, time_point=time_points)
                         for modality in modality_indices}
            array_list = []
            # Iterate over the dictionary and convert each value to a numpy array
            for key in augm_imgs:
                array_list.append(np.array(augm_imgs[key]))
            # Stack the list of numpy arrays into a single numpy array
            final_array = np.stack(array_list)
            result = np.transpose(final_array, (1, 0) + tuple(range(2, final_array.ndim)))
            self.data[int(self.counter_patient)] = result
            self.counter_patient += 1

    def generate_gaussian(self, shape, center, std_dev):
        """
        Generate a Gaussian distribution with the same dimensionality as the input shape.

        Parameters:
        shape (tuple): The shape of the target array (e.g., (240, 240) or (240, 240, 3)).
        center (tuple): The center of the Gaussian distribution (e.g., (120, 120)).
        std_dev (float): The standard deviation of the Gaussian distribution.

        Returns:
        numpy.ndarray: A 2D or 3D array with the Gaussian distribution.
        """
        # Create coordinate grids based on the shape
        if len(shape) == 2:
            y, x = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), indexing='ij')
            gaussian_2d = np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / (2 * std_dev ** 2))
            return gaussian_2d

        elif len(shape) == 3:
            y, x = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), indexing='ij')
            gaussian_2d = np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / (2 * std_dev ** 2))

            # Repeat the 2D Gaussian across the third dimension
            gaussian_3d = np.repeat(gaussian_2d[:, :, np.newaxis], shape[2], axis=2)
            return gaussian_3d

        else:
            raise ValueError("Input shape must be 2D or 3D.")

    def deform_structure(self, modality, context_img, label_img, intensity=20, time_point=None, growth_max=5,
                         apply_prob=0.25):
        # so we do not waste compute
        dil_magnitude = intensity * np.random.uniform(2, growth_max)
        if np.random.rand() < 0.5:
            dil_magnitude *= -1
        loc_index = np.argmax(np.sum(label_img, axis=(0, 1)))
        # context_img = context_img
        dim = context_img.shape
        tmp = [np.arange(i) for i in
               dim]  # return evenly spaced values between 0 and shape of dimension for all dimensions
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)  # (3, 240, 240, 155)
        for d in range(len(dim)):
            coords[d] -= ((np.array(dim).astype(float) - 1) / 2.)[d]
            # zero centers the mesh (origin moved to center of image)
        # save coords for now
        coord_save = coords.copy()
        # organ gradient field
        spacing_ratio, blur = 1.0, 3
        organ_blurred = gaussian_filter((label_img >= 1).astype(float), sigma=(blur * spacing_ratio, blur, blur),
                                        order=0,
                                        mode='nearest')
        t, u, v = np.gradient(organ_blurred)  # x,y,z == h,w,d
        if np.random.rand() < apply_prob:
            vec_metric = ((u ** 2 + v ** 2 + t ** 2) ** 0.5)[:, :, :]
            # Get the indices of non-zero elements
            non_zero_indices = np.array(np.nonzero(vec_metric > 0.1)).T
            # Randomly select 2 points from these indices
            random_point = non_zero_indices[np.random.choice(non_zero_indices.shape[0], 1, replace=False)]
            gauss = self.generate_gaussian(vec_metric.shape, random_point[0], 25)
        else:
            gauss = 1

        if np.random.rand() < apply_prob:
            if self.memory_efficient:
                # we do not want a value into the direction that is zero anyways
                vec = np.append(np.random.rand(2), 0)
            else:
                vec = np.random.rand(3)
            v_hat = vec / np.linalg.norm(vec)
        else:
            v_hat = [1, 1, 1]

        # dil_magnitude = intensity  # amplitude ('height') of the gaussian
        # add a time dependent component to the gradient field
        if time_point is None:
            # time_point = np.random.choice(time_points)
            coords[0, :, :, :] += t * dil_magnitude
            coords[1, :, :, :] += u * dil_magnitude
            coords[2, :, :, :] += v * dil_magnitude * spacing_ratio
            for d in range(3):
                ctr = context_img[0].shape[d] / 2
                coords[d] += ctr - 0.5
            return map_coordinates(context_img[:, :, :], coords, order=1, mode='nearest')
        else:
            res_list = []
            # had to rename
            for time_point_time in time_point:
                coords = coord_save.copy()
                coords[0, :, :, :] += t * dil_magnitude * gauss * v_hat[0] * time_point_time
                coords[1, :, :, :] += u * dil_magnitude * gauss * v_hat[0] * time_point_time
                coords[2, :, :, :] += v * dil_magnitude * gauss * v_hat[0] * time_point_time * spacing_ratio
                # We do note rotate for now
                if np.random.rand() < 0.00:
                    coords = rotate_coords_3d(coords, np.random.uniform(0, 0.1), np.random.uniform(0, 0.1),
                                              np.random.uniform(0, 0.1))
                for d in range(3):
                    ctr = context_img.shape[d] / 2
                    coords[d] += ctr - 0.5
                res_loc = map_coordinates(context_img[:, :, :], coords, order=1, mode='nearest')
                # i.e. we would like to reshape this into a smaller tensor!
                if self.two_dim:
                    new_dims = []
                    for original_length, new_length in zip(res_loc.shape[:2], (128, 128)):
                        new_dims.append(np.linspace(0, original_length - 1, new_length))
                        # new_shape: 96x96x64?
                    # Step 3: Create a meshgrid of the new coordinates
                    coords_down = np.meshgrid(*new_dims, indexing='ij')

                    # Step 4: Interpolate the first two dimensions of the tensor to the new shape
                    # Select the slice from the last dimension (e.g., slice index 75)
                    # check whether label at the following depth is zero, if so check for a different depth
                    if np.sum(res_loc[..., loc_index]) == 0:
                        print('Changed Label is zero at depth: ', loc_index)
                        # find the next non zero depth
                        # calculate the maximum label size?
                        loc_index = np.argmax(np.sum(res_loc, axis=(0, 1)))
                    res_loc = map_coordinates(res_loc[..., loc_index], coords_down)
                    if res_loc.sum() == 0:
                        print('still zero')
                else:
                    new_dims = []
                    for original_length, new_length in zip(res_loc.shape, (96, 96, 64)):
                        new_dims.append(np.linspace(0, original_length - 1, new_length))

                    # Step 3: Create a meshgrid of the new coordinates
                    coords_down = np.meshgrid(*new_dims, indexing='ij')
                    new_coords = np.stack([g.ravel() for g in coords_down], axis=-1)

                    # Step 4: Use RegularGridInterpolator to downsample
                    original_grid = (
                        np.linspace(0, res_loc.shape[0] - 1, res_loc.shape[0]),
                        np.linspace(0, res_loc.shape[1] - 1, res_loc.shape[1]),
                        np.linspace(0, res_loc.shape[2] - 1, res_loc.shape[2])
                    )
                    interpolator = RegularGridInterpolator(original_grid, res_loc, method='linear')
                    res_loc = interpolator(new_coords).reshape(96, 96, 64)

                    # Check if the result is zero
                    if res_loc.sum() == 0:
                        print("Downsampled result is zero.")
                        print('still zero, something went very wrong')
                res_list.append(res_loc)

            # should be T,C,H,W,D
            final_res = np.stack(res_list)
            return final_res

    def load_single_data(self, index):
        return self.data[index]

    def _get_data_shape(self):
        # get data shape from one sample
        sample = self.data[0]
        T,C,D,H,W = sample.shape # total shape
        return (int(T-1), C, D, H, W)


if __name__ == "__main__":
    print('Debugging the growth script')
    data_set = SemiSynthLongi()
    next(iter(data_set))
    # data_set.generate_save_data()
    # data_set.augment_data()

    '''
            elif args.dataset == 'brats':
            from .brats_semi_loader import SemiSynthLongi #TODO: rename to something more general if we use it for non-Brats data
            # todo: also import it
            print("Currently not yet supported. ")
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = SemiSynthLongi(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
    '''