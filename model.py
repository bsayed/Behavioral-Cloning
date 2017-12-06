import argparse
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocessing import *


# Generator used to train the model
def generator(samples, imgs_dir, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)  # Shuffle training data
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = imgs_dir + batch_sample[0].split('/')[-1]
                img_left = imgs_dir + batch_sample[1].split('/')[-1]
                img_right = imgs_dir + batch_sample[2].split('/')[-1]

                # Convert the steering angle from string to float value
                center_angle = float(batch_sample[3])

                # Create adjusted steering measurements for the side camera images
                correction = 0.25  # This value was chosen empirically
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # Pick a number randomly between [0, 3) to use to train the model with either center, left, or
                # right image, this reduce the effect of bias towards driving straight.
                rnd_num = np.random.randint(0, 3)

                if rnd_num == 0:
                    # Read the image from filesystem
                    right_image = cv2.imread(img_right)
                    # Pre-process and augment the image
                    selected_img, selected_angle = preprocess_img(right_image, steering_right)
                elif rnd_num == 1:
                    # Read the image from filesystem
                    left_image = cv2.imread(img_left)
                    # Pre-process and augment the image
                    selected_img, selected_angle = preprocess_img(left_image, steering_left)
                else:
                    # Read the image from filesystem
                    center_image = cv2.imread(img_center)
                    # Pre-process and augment the image
                    selected_img, selected_angle = preprocess_img(center_image, center_angle)

                images.append(selected_img)
                angles.append(selected_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Model')
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file that contains the paths for training images.'
    )
    parser.add_argument(
        'images_dir',
        type=str,
        help='Path to directory containing training images.'
    )
    parser.add_argument(
        'model_file',
        type=str,
        default='model.h5',
        nargs='?',
        help='Path to file where the model will be stored, default="model.h5".'
    )
    parser.add_argument(
        'batch_size',
        type=int,
        default=32,
        nargs='?',
        help='Batch size.'
    )
    parser.add_argument(
        'learning_rate',
        type=float,
        default=1e-4,
        nargs='?',
        help='Learning rate.'
    )
    parser.add_argument(
        'epochs',
        type=int,
        default=10,
        nargs='?',
        help='Number of Epochs.'
    )

    args = parser.parse_args()

    row, col, ch = 64, 64, 3  # image shape 64x64 by 3 channels RGB

    samples = []
    with open(args.csv_file) as csv_file:
        reader = csv.reader(csv_file)

        # ignoring the first line (header)
        next(reader, None)

        for line in reader:
            samples.append(line)

    # split the data into train and validation samples with ratio 80% for training and 20% for validation
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print("Train Samples size: {}".format(len(train_samples)))
    print("Validation Samples size: {}".format(len(validation_samples)))

    # early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    file_path = args.model_file
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # compile and train the model using the generator function
    train_generator = generator(train_samples, imgs_dir=args.images_dir, batch_size=args.batch_size)
    validation_generator = generator(validation_samples, imgs_dir=args.images_dir, batch_size=args.batch_size)

    # The model is based on Nvidia's paper "End to End learning for self-driving cars"
    model = Sequential()
    # Pre-process incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))

    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(50))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(1))

    adam_opt = Adam(args.learning_rate)

    model.summary()

    model.compile(loss='mse', optimizer=adam_opt)
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), callbacks=callbacks_list, nb_epoch=args.epochs)

    model.save('model.h5')
