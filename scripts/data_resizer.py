import os

from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import subfiles


def main():
    PATH = "/Users/hack-machine/School/smad/all_data/wound-segmentation/Foot Ulcer Segmentation Challenge"
    data_types = ["images", "labels"]
    data_purpose = ["test", "train"]
    im_format = ".png"
    final_path = "/Users/hack-machine/School/smad/all_data/resized_data/" + PATH.split("/")[-1]
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    for purpose in data_purpose:
        if not os.path.exists(final_path + "/" + purpose):
            os.makedirs(final_path + "/" + purpose)
        for types in data_types:
            if not os.path.exists(final_path + "/" + purpose + "/" + types):
                os.makedirs(final_path + "/" + purpose + "/" + types)
            try:
                for file in subfiles(PATH + "/" + purpose + "/" + types, suffix=im_format, join=False):
                    im = Image.open(PATH + "/" + purpose + "/" + types + "/" + file)
                    im.resize((640, 480))

                    im.save(final_path + "/" + purpose + "/" + types + "/" + file)
            except FileNotFoundError as e:
                print(e)
                break


if __name__ == "__main__":
    main()
