import cv2
import glob
import numpy
import tensorflow as tf
from PIL import Image
from mixup_utils import mix_up
import os


def tensor_to_image(tensor, to_black_white=False):
    tensor = tensor * 255
    tensor = numpy.array(tensor, dtype=numpy.uint8)

    if numpy.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    if to_black_white:
        tensor[tensor > 0] = 255

    return Image.fromarray(tensor)


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    return int(int_part)


def do_mixup_from_to_imgs(from_nth_img: int, to_nth_img: int):
    global im, output_name, AUTO, idx, label
    x_train = []
    y_train = []
    # get np arr of tr input
    counter = 0
    for input_name in sorted(glob.glob(
            '/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/input/*jpg'),
            key=get_key):
        counter += 1
        if counter <= from_nth_img:
            continue
        im = cv2.imread(input_name)
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x_train.append(RGB_img)
        if counter == to_nth_img:
            break
    counter = 0
    for output_name in sorted(glob.glob(
            '/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/output/*png'),
            key=get_key):
        counter += 1
        if counter <= from_nth_img:
            continue
        im = cv2.imread(output_name)
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        RGB_img[RGB_img < 255] = 0
        y_train.append(RGB_img)
        if counter == to_nth_img:
            break
    # to np arrs

    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    x_train = x_train.astype("float32") / 255.0
    y_train = y_train.astype("float32") / 255.0
    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = 1600
    train_ds_one = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BATCH_SIZE).batch(BATCH_SIZE))
    train_ds_two = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BATCH_SIZE).batch(BATCH_SIZE))
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    # First create the new dataset using our `mix_up` utility
    train_ds_mu = train_ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=1), num_parallel_calls=AUTO)
    sample_images, sample_labels = next(iter(train_ds_mu))
    idx = from_nth_img + 1
    for (img, label) in zip(sample_images, sample_labels):
        img = tensor_to_image(img)
        label = tensor_to_image(label, to_black_white=True).convert('L')
        img.save(
            f"/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/input_mixup/{idx+2050}.jpg")
        label.save(
            f"/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/output_mixup/{idx+2050}.png")
        idx += 1


if __name__ == "__main__":
    # quick workaround to do augmentations in 'batches'
    do_mixup_from_to_imgs(0, 400)
    do_mixup_from_to_imgs(400, 800)
    do_mixup_from_to_imgs(800, 1200)
    do_mixup_from_to_imgs(1200, 1600)
    do_mixup_from_to_imgs(1600, 2000)
