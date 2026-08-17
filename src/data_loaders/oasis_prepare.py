"""
Prepares raw OASIS-2 into the processed/<subj>_<d|nd>/visit_NN/ layout that
oasis_loader.py's OASISDataset expects, including the real FSL-based segmentation
targets (FAST tissue segmentation for CSF, FIRST for hippocampus) that make
target_seg/target_seg_csf/target_seg_hipp real anatomical ROI supervision instead
of an all-zero placeholder.

Per visit, this script:
    1. Converts the raw Analyze (.img/.hdr) T1 to NIfTI.
    2. Brain-extracts it with hd-bet.
    3. Registers the brain-extracted T1 to MNI152 space with FLIRT.
    4. Runs FSL FAST for a 3-tissue (CSF/GM/WM) segmentation of the registered T1.
    5. Runs FSL run_first_all (hippocampus only) for a FIRST-based subcortical
       segmentation, giving standard label IDs 17=L_Hipp/53=R_Hipp.

Requires FSL (FAST, FLIRT, run_first_all, fslchfiletype) with a working FSLDIR —
see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki for installation — and hd-bet
(pip install hd-bet) for brain extraction. Neither is a plain pip dependency of
this repo; both are only needed to run this prep script, not for training/eval.

Data access: OASIS-2 is distributed under its own data-use agreement, not a plain
public download — register at https://www.oasis-brains.org/ to request access.
The longitudinal demographics spreadsheet (oasis_longitudinal_demographics-*.xlsx)
is provided from the same source.

Expected raw layout (as OASIS-2 ships it):
    <DATA_DIR>/oasis/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx
    <DATA_DIR>/oasis/OAS2_RAW_PART1/<MRI ID>/RAW/mpr-1.nifti.img (+ .hdr)
    <DATA_DIR>/oasis/OAS2_RAW_PART2/<MRI ID>/RAW/mpr-1.nifti.img (+ .hdr)

Usage:
    python src/data_loaders/oasis_prepare.py --exec   # actually run; omit for a dry run
"""
import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "./data/")
METADATA_XLSX = os.path.join(DATA_DIR, "oasis", "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx")
RAW_ROOTS = [
    os.path.join(DATA_DIR, "oasis", "OAS2_RAW_PART1"),
    os.path.join(DATA_DIR, "oasis", "OAS2_RAW_PART2"),
]
PROCESSED_ROOT = os.path.join(DATA_DIR, "oasis", "processed")

FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
os.environ["FSLDIR"] = FSLDIR
MNI_BRAIN = os.path.join(FSLDIR, "data/standard/MNI152_T1_1mm_brain.nii.gz")

DRY_RUN = True  # safety default; only set False via --exec


def run(cmd):
    print("CMD:", " ".join(str(c) for c in cmd))
    if not DRY_RUN:
        subprocess.run(cmd, check=True)


def find_raw_dir(visit_id: str):
    for root in RAW_ROOTS:
        candidate = Path(root) / visit_id / "RAW"
        if candidate.is_dir():
            return candidate
    return None


def get_group_flag(group_str: str) -> str:
    g = str(group_str).strip().lower()
    if "non" in g:
        return "nd"
    if "demented" in g:
        return "d"
    return "unk"


def process_visit(subj_id, mri_id, group, visit_num, age):
    raw_dir = find_raw_dir(mri_id)
    if raw_dir is None:
        print(f"[WARN] RAW dir not found for {mri_id}, skipping.\n")
        return
    in_img = raw_dir / "mpr-1.nifti.img"
    if not in_img.exists():
        print(f"[WARN] Missing mpr-1.nifti.img for {mri_id}, skipping.\n")
        return

    subj_tag = f"{subj_id}_{get_group_flag(group)}"
    visit_tag = f"visit_{int(visit_num):02d}"
    out_dir = Path(PROCESSED_ROOT) / subj_tag / visit_tag
    if out_dir.exists():
        print(f"[INFO] Output dir already exists: {out_dir}, skipping.\n")
        return
    print(f"OUT DIR: {out_dir}")
    if not DRY_RUN:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Convert Analyze -> NIfTI ---
    t1_prefix = out_dir / "t1_raw"
    run(["fslchfiletype", "NIFTI_GZ", str(in_img), str(t1_prefix)])
    t1_nii = str(t1_prefix) + ".nii.gz"

    # --- 2. hd-bet brain extraction (writes <out>.nii.gz + <out>_mask.nii.gz) ---
    brain_nii = out_dir / "t1_raw_bet.nii.gz"
    run(["hd-bet", "-i", t1_nii, "-o", str(brain_nii), "-device", "cpu", "-mode", "fast"])

    # --- 3. FLIRT registration to MNI152 (use the brain-extracted volume) ---
    reg_nii = out_dir / "t1_brain_MNI.nii.gz"
    mat_file = out_dir / "t1_to_MNI.mat"
    run(["flirt", "-in", str(brain_nii), "-ref", MNI_BRAIN,
         "-out", str(reg_nii), "-omat", str(mat_file), "-dof", "12"])

    # --- 4. FSL FAST: 3-tissue segmentation (0=bg, 1=CSF, 2=GM, 3=WM) of the
    #    registered brain, giving target_seg_csf its real CSF/ventricle mask. ---
    seg_prefix = out_dir / "t1_seg"
    run(["fast", "-o", str(seg_prefix), "-t", "1", str(reg_nii)])

    # --- 5. FSL run_first_all: hippocampus-only FIRST segmentation, giving
    #    target_seg_hipp its real L_Hipp/R_Hipp mask (standard label IDs 17/53). ---
    first_prefix = out_dir / "t1_first_hipp"
    run(["run_first_all", "-i", str(reg_nii), "-o", str(first_prefix), "-s", "L_Hipp,R_Hipp"])

    print("DONE (dry run)" if DRY_RUN else "DONE (executed)")
    print()


def main():
    print("DRY_RUN =", DRY_RUN)
    print("No commands will be executed." if DRY_RUN else "Commands WILL be executed.")
    assert os.path.isfile(METADATA_XLSX), (
        f"{METADATA_XLSX} not found — see this file's docstring for the expected "
        f"raw OASIS-2 layout and data-access process."
    )
    df = pd.read_excel(METADATA_XLSX)

    for _, row in df.iterrows():
        subj_id = str(row["Subject ID"]).strip()
        mri_id = str(row["MRI ID"]).strip()
        if "MR" not in mri_id:
            continue
        group = row["Group"]
        visit_num = int(row["Visit"])
        age = int(row["Age"]) if not pd.isna(row["Age"]) else -1

        print(f"=== {subj_id} | {mri_id} | {group} | visit {visit_num} | age {age} ===")
        process_visit(subj_id, mri_id, group, visit_num, age)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OASIS-2 preprocessing script")
    parser.add_argument("--exec", action="store_true", help="Actually run the pipeline (default: dry run, print only).")
    args = parser.parse_args()
    DRY_RUN = not args.exec
    main()
