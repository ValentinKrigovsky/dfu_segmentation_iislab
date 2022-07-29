import os

import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import subfiles
import torchvision.transforms.functional as F


def change_hue(image):
    changed_hue = F.adjust_hue(image, -0.1)
    return changed_hue


def main():
    PATH = "/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/"
    data_types = ["input", "output"]
    new_data_types = ["input_hue", "output_hue"]
    im_format = ".jpg"
    final_path = "/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all"

    if not os.path.exists(final_path):
        os.makedirs(final_path)

    for data_type in new_data_types:
        if not os.path.exists(final_path + "/" + data_type):
            os.makedirs(final_path + "/" + data_type)

    for file in subfiles(PATH + data_types[0], suffix=im_format, join=False):
        im = Image.open(PATH + data_types[0] + "/" + file)

        changed_image = F.adjust_hue(im, -0.1)

        changed_image.save(final_path + "/" + new_data_types[1] + "/" + file)

        print(file)
        mask = Image.open(PATH + data_types[1] + "/" + file.split(".")[0] + ".png")
        mask.save(final_path + "/" + new_data_types[0] + "/" + file.split(".")[0] + ".png")


if __name__ == "__main__":
    main()
