import tensorflow as tf

IMG_W = 640
IMG_H = 480


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = IMG_W * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_H * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_W, dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_H, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_W)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_H)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_W)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_H)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


@tf.function
def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, IMG_H, IMG_W
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, IMG_H, IMG_W
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Get a patch from the second label (`label2`)
    crop_label2 = tf.image.crop_to_bounding_box(
        label2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `label2` patch (`crop_label2`) with the same offset
    label2 = tf.image.pad_to_bounding_box(
        crop_label2, boundaryy1, boundaryx1, IMG_H, IMG_W
    )
    # Get a patch from the first label (`label1`)
    crop_label1 = tf.image.crop_to_bounding_box(
        label1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `label1` patch (`crop_label1`) with the same offset
    msk1 = tf.image.pad_to_bounding_box(
        crop_label1, boundaryy1, boundaryx1, IMG_H, IMG_W
    )

    # Modify the first label by subtracting the patch from `label1`
    # (before applying the `label2` patch)
    label1 = label1 - msk1
    # Add the modified `label1` and `label2`  together to get the CutMix label
    label = label1 + label2

    return image, label
