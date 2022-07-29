import os

import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import subfiles


def calc_whites(image):
    d = image.getdata()
    white_pix = 0
    for pixel in d:
        if pixel == 255:
            white_pix += 1
    return white_pix


def resize_n_crop_image(image):
    height = 480
    width = 640
    new_h = height * 2
    new_w = width * 2
    resized_im = image.resize((new_w, new_h))
    # resized_im.save("/Users/hack-machine/School/diplo/test/new/kokotina.jpg")

    left = (new_w / 2) - 320
    top = (new_h / 2) - 240
    right = (new_w / 2) + 320
    bottom = (new_h / 2) + 240
    hh = bottom - top
    ww = right - left

    cropped = resized_im.crop((left, top, right, bottom))
    return cropped


def main():
    PATH = "/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/"
    data_types = ["input_cutmix", "output_cutmix"]
    new_data_types = ["input_mixup_zoomed2x", "output_mixup_zoomed2x"]
    im_format = ".png"
    final_path = "/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all"
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    for data_type in new_data_types:
        if not os.path.exists(final_path + "/" + data_type):
            os.makedirs(final_path + "/" + data_type)

    for file in subfiles(PATH + data_types[1], suffix=im_format, join=False):
        im = Image.open(PATH + data_types[1] + "/" + file)

        old_w = calc_whites(im)

        # dlzka = len(d)
        # white_ratio = (white_pix/len(d)) * 1000

        cropped_im = resize_n_crop_image(im)
        cropped_w = calc_whites(cropped_im)

        # TODO: find better coeficient
        # TODO: better zoom, better coeficient
        if cropped_w > 2 * old_w:
            print(f"Created photo: {file}.")
            pixels = cropped_im.load()  # create the pixel map

            for i in range(cropped_im.size[0]):  # for every pixel:
                for j in range(cropped_im.size[1]):
                    if pixels[i, j] != 0:
                        # change to black if not red
                        pixels[i, j] = 255
            cropped_im.save(final_path + "/" + new_data_types[1] + "/10" + file)

            originales = Image.open(PATH + data_types[0] + "/" + file.split(".")[0] + ".jpg")
            originales = resize_n_crop_image(originales)
            originales.save(final_path + "/" + new_data_types[0] + "/10" + file.split(".")[0] + ".jpg")


if __name__ == "__main__":
    main()
