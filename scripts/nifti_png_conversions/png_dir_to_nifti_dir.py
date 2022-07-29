import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

if __name__ == '__main__':

    base = '/home/dfu/first_nnu-net_experiments/challenge_data/DFUC2022_test_release'

    # now start the conversion
    target_base = join(f'{base}_niftis')

    maybe_mkdir_p(target_base)
    cases = subfiles(base, join=False)

    for t in cases:
        unique_name = t[:-4]
        # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_image_file = join(base, f'{unique_name}.jpg')

        output_image_file = join(target_base, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. Youcdon't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)