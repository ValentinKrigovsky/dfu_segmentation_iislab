import cv2
import glob
import numpy
import tensorflow as tf
from PIL import Image
from cutmix_utils import cutmix
import os


def preprocess_image(image, label):
    return image, label


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    return int(int_part)


def tensor_to_image(tensor, to_black_white=False):
    tensor = tensor * 255
    tensor = numpy.array(tensor, dtype=numpy.uint8)
    if numpy.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    img = Image.fromarray(tensor)
    should_save = False
    if to_black_white:
        img = img.convert('L')
        if numpy.sum(tensor > 0) > 5000:
            should_save = True
    return img, should_save


def method_name(from_nth_img: int, to_nth_img: int):
    global counter, im, output_name, AUTO, idx, label, filename
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
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        print(output_name)
        y_train.append(RGB_img)
        if counter == to_nth_img:
            break
    # to np arrs
    print('here')
    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)
    x_train = x_train.astype("float32") / 255.0
    y_train = y_train.astype("float32") / 255.0
    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = 810
    train_ds_one = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(BATCH_SIZE) \
        .map(preprocess_image, num_parallel_calls=AUTO)
    train_ds_two = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(BATCH_SIZE) \
        .map(preprocess_image, num_parallel_calls=AUTO)
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    train_ds_cmu = train_ds.shuffle(BATCH_SIZE).map(cutmix, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
    sample_images, sample_labels = next(iter(train_ds_cmu))
    idx = from_nth_img + 1
    for (img, label) in zip(sample_images, sample_labels):
        img, should_save_img = tensor_to_image(img)
        label, should_save_label = tensor_to_image(label, to_black_white=True)

        filename = f"{idx}.png"
        if should_save_label:
            img.save(
                f"/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/input_cutmix/{idx}.jpg")
            label.save(
                f"/home/dfu/first_nnu-net_experiments/foot_ulcer_unconverted_png_data/DFU2022_ready_split/all/output_cutmix/{idx}.png")

        idx += 1

if __name__ == "__main__":
    method_name(0, 600)
    method_name(600, 1200)
    method_name(1200, 1600)
    method_name(1600, 2000)
