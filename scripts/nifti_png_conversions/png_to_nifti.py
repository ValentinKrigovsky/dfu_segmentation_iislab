import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

if __name__ == '__main__':

    # extract the zip file, then set the following path according to your system:
    base = '/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split'
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task526_mixup_cutmix_255_zoomed2'
    target_base = join(nnUNet_raw_data, task_name)
    maybe_mkdir_p(target_base)
    print(target_base)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir_tr = join(base, 'all', 'output')
    images_dir_tr = join(base, 'all', 'input')
    training_cases = subfiles(labels_dir_tr, join=False)

    for t in training_cases:
        unique_name = t[:-4]
        # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, f'{unique_name}.jpg')

        output_image_file = join(target_imagesTr,
                                 'dfu_' + unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr,
                               'dfu_' + unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. Youcdon't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))
    # commented out because we use all data
    # # now do the same for the test set
    # labels_dir_ts = join(base, 'testing', 'output')
    # images_dir_ts = join(base, 'testing', 'input')
    # testing_cases = subfiles(labels_dir_ts, join=False)
    # for ts in testing_cases:
    #     unique_name = ts[:-4]
    #     input_segmentation_file = join(labels_dir_ts, ts)
    #     input_image_file = join(images_dir_ts,  f'{unique_name}.jpg')
    #
    #     output_image_file = join(target_imagesTs, 'dfu_' + unique_name)
    #     output_seg_file = join(target_labelsTs, 'dfu_' + unique_name)
    #
    #     convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
    #     convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
    #                               transform=lambda x: (x > 0).astype(int))

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Red', 'Green', 'Blue'),
                          labels={0: 'background', 1: 'ulcer'}, dataset_name=task_name, license='hands off!')
