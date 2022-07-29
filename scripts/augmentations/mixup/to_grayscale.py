import glob
import numpy
import tensorflow as tf
from PIL import Image
from mixup_utils import mix_up

if __name__ == "__main__":

    x_train = []

    for output_name in glob.glob('../../../training_cutmix_mixedup/output/*png'):
        filename = output_name.split('\\')[1]
        img = Image.open(output_name).convert('L')
        img.save(f'../../../training_cutmix_mixedup/output/{filename}')
