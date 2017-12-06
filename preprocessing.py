import numpy as np
import cv2
from scipy import ndimage


def random_shift(img, shift, angle):
    """Translate image in x direction randomly"""
    x_shift = np.random.uniform(-shift, shift)
    # y_shift = np.random.uniform(-shift, shift)
    new_angle = angle + (-x_shift * 0.004)  # 0.004 angle change for every pixel shifted

    return ndimage.shift(img, (0, x_shift, 0)), new_angle


def random_brightness(image):
    """Apply random brightness on HSV formatted image"""
    # new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    new_img = np.array(image)
    random_bright = .5 + np.random.uniform()
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright
    new_img[:, :, 2][new_img[:, :, 2] > 255] = 255
    new_img = np.array(new_img)

    # new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

    return new_img


def random_saturation(image):
    """Apply random saturation on HSV formatted image"""
    saturation_threshold = 0.4 + 1.2 * np.random.uniform()
    new_img = np.array(image)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    new_img[:, :, 1] = new_img[:, :, 1] * saturation_threshold
    new_img = np.array(new_img)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img


def random_shadow(image):
    top_y = image.shape[1] * np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1] * np.random.uniform()
    # image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_hls = image
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    # new_img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    new_img = image_hls

    return new_img


def random_lightness(image):
    lightness_threshold = 0.2 + 1.4 * np.random.uniform()
    new_img = np.array(image)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HLS)
    new_img[:, :, 1] = new_img[:, :, 1] * lightness_threshold
    # new_img = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

    return new_img


def preprocess_img(image, angle):
    """Preprocess and augment the image"""

    new_angle = angle

    # Crop image, 60 pixels from top and 25 from bottom
    new_img = image[60:135, :, :]

    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    # Resize image to 200 width and 66 height
    new_img = cv2.resize(new_img, (64, 64), interpolation=cv2.INTER_AREA)

    if flip_a_coin():
        # Apply blurring
        new_img = cv2.GaussianBlur(new_img, (3, 3), 0)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    if flip_a_coin():
        # Apply random brightness
        new_img = random_brightness(new_img)

    if flip_a_coin():
        # Apply random saturation
        new_img = random_saturation(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HLS)
    if flip_a_coin():
        # Apply random lightness
        new_img = random_lightness(new_img)

    if flip_a_coin():
        # Apply random shadow
        new_img = random_shadow(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HLS2RGB)

    if flip_a_coin():
        # Apply random translation in the x-axis and get new image and new angle
        new_img, new_angle = random_shift(new_img, 25, angle)

    if flip_a_coin():
        # Flip image and steering angle
        new_img = np.fliplr(new_img)
        new_angle = -new_angle

    return new_img, new_angle


def flip_a_coin():
    return np.random.uniform() <= .5
