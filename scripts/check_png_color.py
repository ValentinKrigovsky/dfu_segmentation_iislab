import os
from PIL import Image
data_directory = "/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/training/output"

for filename in os.listdir(data_directory):
    img = Image.open(data_directory + '/' + filename)
    extrema = img.convert("L").getextrema()
    # print(extrema)
    if extrema != (0, 255): #Black map
        print(filename)
    #     shutil.copyfile(photo_path, class_A_folder)
    # else:
    #     shutil.copyfile(photo_path, class_B_folder)

