import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Convolution2D, AveragePooling2D, MaxPooling2D, \
    Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import cv2
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from scipy import ndimage

row, col, ch = 160, 320, 3  # image format

samples = []
with open('../fourthSimData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # ignoring the first line (header)
    next(reader, None)

    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Train Samples size: {}".format(len(train_samples)))
print("Validation Samples size: {}".format(len(validation_samples)))


def shift_random(img, shift):
    """Translate image in x and y direction"""
    x_shift = np.random.uniform(-shift, shift)
    # y_shift = np.random.uniform(-shift, shift)
    return ndimage.shift(img, (x_shift, 0, 0), mode='nearest')


def preprocess_img(image):
    new_img = image[50:140, :, :]
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    new_img = shift_random(new_img, 20)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    new_img = np.array(new_img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright
    new_img[:, :, 2][new_img[:, :, 2] > 255] = 255
    new_img = np.array(new_img, dtype=np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img


def transform_images_in_line(line, write_path, read_path):
    img_center_path = read_path + line[0].split('/')[-1]
    img_left_path = read_path + line[1].split('/')[-1]
    img_right_path = read_path + line[2].split('/')[-1]

    steering_angle = line[3]
    throttle = line[4]

    aug_lines = []
    # create 10 images with random brightness for each of the center, left, and right image
    for i in range(5):
        c_img = preprocess_img(cv2.imread(img_center_path))
        c_img_path = write_path + line[0].split('/')[-1].split('.')[0] + "_{}".format(i) + ".jpg"
        cv2.imwrite(c_img_path, c_img)
        l_img = preprocess_img(cv2.imread(img_left_path))
        l_img_path = write_path + line[1].split('/')[-1].split('.')[0] + "_{}".format(i) + ".jpg"
        cv2.imwrite(l_img_path, l_img)
        r_img = preprocess_img(cv2.imread(img_right_path))
        r_img_path = write_path + line[2].split('/')[-1].split('.')[0] + "_{}".format(i) + ".jpg"
        cv2.imwrite(r_img_path, r_img)

        aug_lines.append([c_img_path, l_img_path, r_img_path, steering_angle, throttle])

    return aug_lines


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = '../fourthSimData/IMG/' + batch_sample[0].split('/')[-1]
                img_left = '../fourthSimData/IMG/' + batch_sample[1].split('/')[-1]
                img_right = '../fourthSimData/IMG/' + batch_sample[2].split('/')[-1]

                # Code for Amazon EC2
                center_image = cv2.imread(img_center)
                # center_image = preprocess_img(center_image)
                # center_image = cv2.resize(center_image, (200, 66), interpolation=cv2.INTER_AREA)

                left_image = cv2.imread(img_left)
                # left_image = preprocess_img(left_image)
                # left_image = cv2.resize(left_image, (200, 66), interpolation=cv2.INTER_AREA)

                right_image = cv2.imread(img_right)
                # right_image = preprocess_img(right_image)
                # right_image = cv2.resize(right_image, (200, 66), interpolation=cv2.INTER_AREA)

                center_angle = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                right_correction = 0.15  # this is a parameter to tune
                left_correction = 0.15
                steering_left = center_angle + left_correction
                steering_right = center_angle - right_correction

                # Append flipped images
                flipped_image = np.fliplr(center_image)
                flipped_angle = -center_angle

                images.extend([center_image, flipped_image])
                angles.extend([center_angle, flipped_angle])

                # if center_angle < -0.15:
                #     images.extend([right_image])
                #     angles.extend([steering_right])
                # elif center_angle > 0.15:
                #     images.extend([left_image])
                #     angles.extend([steering_left])
                # else:
                #     images.extend([center_image, flipped_image])
                #     angles.extend([center_angle, flipped_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            print(X_train.shape)
            yield shuffle(X_train, y_train)


early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), callbacks=[early_stopping], nb_epoch=7)

model.save('model_commaAI.h5')
