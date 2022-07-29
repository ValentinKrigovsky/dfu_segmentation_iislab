import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_segmentation_nifti_to_img
import pathlib


def extract_num_from_nifti_path(nifti: str):
    return nifti.split('_')[0].split('.')[0]


if __name__ == '__main__':

    niftis_base = '/home/dfu/first_nnu-net_experiments/predictions/val_release_niftis_514_ld'
    pngs_base = f'{niftis_base}/pngs'

    nifti_files = subfiles(niftis_base, suffix='.gz', join=False)

    for nifti_path in nifti_files:
        png_abs_path = f'{pngs_base}/{extract_num_from_nifti_path(nifti_path)}.png'
        nifti_abs_path = f'{niftis_base}/{nifti_path}'

        convert_2d_segmentation_nifti_to_img(nifti_abs_path, png_abs_path, dfu_b_w=True)
