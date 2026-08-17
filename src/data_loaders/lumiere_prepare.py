"""
Prepares raw Lumiere data into the trn_lumiere.npy/val_lumiere.npy/tst_lumiere.npy
files that lumiere_loader.py's LumiereDataset expects (env var DATASET_LOCATION),
extracted out of what used to be inline logic in LumiereDataset.__init__/
generate_save_data (not a standalone, independently-runnable script before this).

Data access: Lumiere is a public longitudinal glioma MRI dataset — see the
official Lumiere dataset release for download and citation details.

Expected raw layout (as Lumiere ships it):
    <DATASET_LOCATION>/lumiere/Lumiere/patients.json
    <DATASET_LOCATION>/lumiere/Lumiere/images_registered/<Patient-Timepoint>/<Patient-Timepoint>_0000.nii.gz
    <DATASET_LOCATION>/lumiere/Lumiere/segmentations_registered/<Patient-Timepoint>.nii.gz

Note: the raw release also ships a ratings.csv (RANO rating per timepoint,
including pre-/post-operative status) that this script does not currently use —
it's there for a possible future pre-op-aware filtering pass, not yet implemented
here (or in the loader).

Usage:
    python src/data_loaders/lumiere_prepare.py
"""
import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from .data_util_functions import filter_and_normalize

DATASET_LOCATION = os.getenv("DATASET_LOCATION", "data/")
DATA_DIR = os.path.join(DATASET_LOCATION, "lumiere", "Lumiere")
RESIZE_SHAPE = (96, 96, 64)


def preprocess(debug: bool = False):
    with open(os.path.join(DATA_DIR, "patients.json")) as f:
        patient_dict = json.load(f)

    images_dir = os.path.join(DATA_DIR, "images_registered")
    segs_dir = os.path.join(DATA_DIR, "segmentations_registered")

    cases_total = {}
    patient_counter = 0
    for patient in patient_dict.keys():
        try:
            patient_time_list = [
                p for p in os.listdir(images_dir) if patient.title().replace("_", "-") in p
            ]
            if debug and patient_counter > 4:
                break

            result_patient_dict = {}
            acquisition_counter = 0
            for patient_time in patient_time_list:
                time = int(patient_time.split("-")[-1].split("_")[0])
                image = nib.load(
                    os.path.join(images_dir, patient_time, f"{patient_time}_0000.nii.gz")
                ).get_fdata()
                segmentation = nib.load(
                    os.path.join(segs_dir, f"{patient_time}.nii.gz")
                ).get_fdata()

                image_resized = zoom(filter_and_normalize(image), zoom=np.divide(RESIZE_SHAPE, image.shape))
                segmentation_resized = zoom(segmentation, zoom=np.divide(RESIZE_SHAPE, segmentation.shape))
                result_patient_dict[acquisition_counter] = image_resized, segmentation_resized, time
                acquisition_counter += 1

            cases_total[patient_counter] = result_patient_dict
            patient_counter += 1
            print(f"Generating data for patient {patient}")
        except Exception:
            print(f"not a valid dir for Patient: {patient}")

    print(f"finished reshaping data for {patient_counter} patients")

    cases = [cases_total[k] for k in sorted(cases_total.keys())]
    # Fixed-count split (matches the original inline logic): first 50 train, next
    # 10 val, remainder test. Not proportional — fine for Lumiere's ~60-80 patients,
    # but would need revisiting if patient count changes substantially.
    np.save(os.path.join(DATASET_LOCATION, "trn_lumiere.npy"), cases[:50], allow_pickle=True)
    np.save(os.path.join(DATASET_LOCATION, "val_lumiere.npy"), cases[50:60], allow_pickle=True)
    np.save(os.path.join(DATASET_LOCATION, "tst_lumiere.npy"), cases[60:], allow_pickle=True)
    print(f"Saved trn ({min(50, len(cases))}), val ({max(0, min(10, len(cases) - 50))}), "
          f"tst ({max(0, len(cases) - 60)}) to {DATASET_LOCATION}")


if __name__ == "__main__":
    preprocess()
