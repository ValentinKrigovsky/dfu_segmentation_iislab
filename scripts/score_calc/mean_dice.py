from monai.metrics import DiceMetric
from monai.metrics import compute_meandice
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.utilities.file_conversions import convert_2d_segmentation_nifti_to_img
import tensorflow as tf
import torch
import argparse
import os
from pydantic import BaseModel
import json
import shutil


class AllPaths(BaseModel):
    original_png_path: str
    original_mask_path: str
    generated_mask_path: str


class WrongPaths(BaseModel):
    exact_0: List[AllPaths]
    bad_performing: List[AllPaths]


def img_path_to_binarized_tensor(path: str):
    convert_tensor = transforms.ToTensor()
    img = Image.open(path)
    tensor = convert_tensor(img).byte()
    tensor[tensor > 0] = 1
    # torch.set_printoptions(profile="full")
    # tf.print(tensor)
    return tensor


def extract_num_from_nifti_path(nifti: str):
    return nifti.split('_')[1].split('.')[0]


def do_meandice(base: str, y_pred_base: str) -> WrongPaths:
    test_filenames = subfiles(y_pred_base, suffix='.png', join=False)
    print(test_filenames)
    all_metrics = []
    wrong_paths = WrongPaths(exact_0=[], bad_performing=[])

    # #just get dice values for exact images
    test_filenames = ['dfu_100215.png', 'dfu_100630.png', 'dfu_100750.png', 'dfu_100835.png']

    # make dice metric
    for filename in test_filenames:
        original_png_path = f'{base}/testing/input/{extract_num_from_nifti_path(filename)}.jpg'
        original_mask_path = f'{base}/testing/output/{extract_num_from_nifti_path(filename)}.png'
        generated_mask_path = f'{y_pred_base}/dfu_{extract_num_from_nifti_path(filename)}.png'
        y = img_path_to_binarized_tensor(original_mask_path)
        y_pred = img_path_to_binarized_tensor(generated_mask_path)

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric(y_pred, y)
        metric = dice_metric.aggregate().item()
        all_metrics.append(metric)
        print(f'{filename}: dice: {metric}')


        # if metric < 0.3:
        #     if metric == 0:
        #         wrong_paths.exact_0.append(
        #             AllPaths(
        #                 original_png_path=original_png_path,
        #                 original_mask_path=original_mask_path,
        #                 generated_mask_path=generated_mask_path))
        #     else:
        #         wrong_paths.bad_performing.append(
        #             AllPaths(
        #                 original_png_path=original_png_path,
        #                 original_mask_path=original_mask_path,
        #                 generated_mask_path=generated_mask_path))

        # if metric > 0.95:
        #     print(f"Image: {original_png_path}, Prediction: {metric}")
        #     wrong_paths.exact_0.append(
        #         AllPaths(
        #             original_png_path=original_png_path,
        #             original_mask_path=original_mask_path,
        #             generated_mask_path=generated_mask_path))

        # if metric == 0:
        #     print(f'{y_base}/{extract_num_from_nifti_path(filename)}.png')

    np_metrics = np.array(all_metrics)
    # non_zero_metrics = np.array(np_metrics[np.where(np_metrics != 0)])
    print(f'mean dice is {np.mean(np_metrics)}')
    # with open(f'{y_pred_base}/mean_dice.txt', 'x') as f:
    #     f.write(f'{np.mean(np_metrics)}')
    return wrong_paths


def main():
    wrong_predictions_dir = "/home/dfu/first_nnu-net_experiments/wrong_predictions"
    new_prediction_dir = wrong_predictions_dir + "/" + args.pred_base.split("/")[-1]
    # zero_dice_path = new_prediction_dir + "/" + "0"
    # other_dice_path = new_prediction_dir + "/" + "other"
    # if not os.path.exists(wrong_predictions_dir):
    #     os.makedirs(wrong_predictions_dir)
    # if not os.path.exists(new_prediction_dir):
    #     os.makedirs(new_prediction_dir)
    #     os.makedirs(zero_dice_path)
    #     os.makedirs(other_dice_path)
        # else:
    #     print(f"ERROR while creating output file. File '{new_prediction_dir}' already exists!")
    #     return

    wrong_dices = do_meandice(
    '/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split',
        '/home/dfu/first_nnu-net_experiments/predictions/512_treshhold255_5050_mixup')
    #
    #
    # # generate files
    for prediction in wrong_dices.exact_0:
        name = prediction.original_png_path.split("/")[-1].split(".")[0]
        path_to_final_dir = new_prediction_dir + "/09/" + name
        if not os.path.exists(path_to_final_dir):
            os.makedirs(path_to_final_dir)
            shutil.copyfile(prediction.original_png_path, path_to_final_dir + '/' + name + ".jpg")
            shutil.copyfile(prediction.original_mask_path,  path_to_final_dir + '/' + name + "_original.png")
            shutil.copyfile(prediction.generated_mask_path,  path_to_final_dir + '/' + name + "_generated.png")
    # for prediction in wrong_dices.bad_performing:
    #     name = prediction.original_png_path.split("/")[-1].split(".")[0]
    #     path_to_final_dir = new_prediction_dir + "/other/" + name
    #     if not os.path.exists(path_to_final_dir):
    #         os.makedirs(path_to_final_dir)
    #         shutil.copyfile(prediction.original_png_path, path_to_final_dir + '/' + name + ".jpg")
    #         shutil.copyfile(prediction.original_mask_path, path_to_final_dir + '/' + name + "_original.png")
    #         shutil.copyfile(prediction.generated_mask_path, path_to_final_dir + '/' + name + "_generated.png")
    #
    # with open("/home/dfu/first_nnu-net_experiments/scripts/score_calc/dice_0.json", "w") as outfile:
    #     outfile.write(json.dumps(wrong_dices.dict(), indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather mean dice values and sort problem photos into dirs')
    parser.add_argument('--base', '-b', type=str,
                        help='a path to tested output pngs')
    parser.add_argument('--pred_base', '-pb', type=str,
                        help='a path to testet output pngs')

    # changed regarding to TINO's comment, default values:
    # y_base = '/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/testing/output'
    # y_pred_base = '/home/dfu/first_nnu-net_experiments/predictions/512_treshhold255_5050_mixup'

    args = parser.parse_args()
    main()
