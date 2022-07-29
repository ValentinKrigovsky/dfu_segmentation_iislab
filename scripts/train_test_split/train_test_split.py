import splitfolders

splitfolders.ratio('/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFUC2022_train_release'
                   '/DFUC2022_train_masks',
                   output="/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFUC2022_train_release"
                          "/DFUC2022_train_masks_split", seed=1337, ratio=(.80, 0, 0.2))
splitfolders.ratio('/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFUC2022_train_release'
                   '/DFUC2022_train_images',
                   output="/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFUC2022_train_release"
                          "/DFUC2022_train_images_split", seed=1337, ratio=(.80, 0, 0.2))