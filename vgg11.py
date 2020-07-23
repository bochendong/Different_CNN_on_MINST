import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import scipy
import PIL
from PIL import Image, ImageFilter
from numpy import random
import skimage
import matplotlib.pyplot as plt
batch_size = 256

def data_argumentation(x_train):
    temp = []
    noise = [0.001, 0.005, 0.01, 0.05，0.1]
    rotations = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    for x in x_train:
        if np.random.random() >= 0.5:
            rand_rot = rotations[random.randint(0, 8)]
            x = np.array(Image.fromarray(x).rotate(rand_rot))

        if np.random.random() >= 0.5:
            rand_var = noise[random.randint(0, 2)]
            x = skimage.util.random_noise(image=x, mode='gaussian', clip=True, mean=0.0, var=rand_var)

        temp.append(np.array(Image.fromarray(x).resize((32, 32))))

    x_train = np.asarray(temp)
    return x_train


def reshape_data(x_train, y_train, x_test, y_test):
    temp = []
    for x in x_train:
        temp.append(np.array(Image.fromarray(x).resize((32, 32))))
    x_train = np.asarray(temp)
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    temp = []
    for x in x_test:
        temp .append(np.array(Image.fromarray(x).resize((32, 32))))
    x_test = np.asarray(temp)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def generate_roubust_data(x_test):
    noise = [0.001, 0.005， 0.01, 0.05，0.1]
    rotations = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    x_test_rotated = [[] for x in range(0, len(rotations))]
    x_test_noise = [[] for x in range(0, len(noise))]

    for i, x in enumerate(x_test):
        x = np.array(Image.fromarray(x).resize((32, 32)))
        for j, rotation in enumerate(rotations):
            x_test_rotated[j].append(np.array(Image.fromarray(x).rotate(rotation)))

    for i, x in enumerate(x_test):
        for j, std in enumerate(noise):
            x = skimage.util.random_noise(
                image=x, mode='gaussian', clip=True, mean=0.0, var=std)
            x = np.array(Image.fromarray(x).resize((32, 32)))

            x_test_noise[j].append(x)

    for i in range(0, len(x_test_rotated)):
        x_test_rotated[i] = np.asarray(x_test_rotated[i])
        x_test_rotated[i] = x_test_rotated[i].reshape(
            x_test_rotated[i].shape[0], 32, 32, 1)

    for i in range(0, len(x_test_noise)):
        x_test_noise[i] = np.asarray(x_test_noise[i])
        x_test_noise[i] = x_test_noise[i].reshape(
            x_test_noise[i].shape[0], 32, 32, 1)

    return (x_test_noise, x_test_rotated)


class VGG_11:
    def __init__(self, dropout=0.5, learning_rate=0.00001):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(4096, activation='relu'))
        if (dropout >= 0 and dropout <= 1):
            model.add(Dropout(dropout))
        model.add(Dense(4096, activation='relu'))
        if (dropout >= 0 and dropout <= 1):
            model.add(Dropout(dropout))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

        self.model = model

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.model.fit(X_train, Y_train, batch_size=256, epochs=5,
                       verbose=1, validation_data=(X_test, Y_test))


def test_VGG_11(data_arg=True, dropout=0.5, learning_rate=0.00001):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if (data_arg == True):
        x_train = data_argumentation(x_train)

    (x_test_noise, x_test_rotated) = generate_roubust_data(x_test)

    (x_train, y_train), (x_test, y_test) = reshape_data(x_train, y_train, x_test, y_test)

    vgg_11_net = VGG_11(dropout=dropout, learning_rate=learning_rate)
    vgg_11_net.fit(x_train, y_train, x_test, y_test)

    rotations = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    noise = [0.001, 0.005， 0.01, 0.05，0.1]

    rotation_loss = []
    rotation_acc = []
    noise_acc = []
    noise_loss = []

    for i, rot in enumerate(rotations):
        score = vgg_11_net.model.evaluate(x_test_rotated[i], y_test, verbose=0)
        rotation_loss.append(score[0])
        rotation_acc.append(score[1])

    for i, std in enumerate(noise):
        score = vgg_11_net.model.evaluate(x_test_noise[i], y_test, verbose=0)
        noise_loss.append(score[0])
        noise_acc.append(score[1])

    return (rotation_loss, rotation_acc, noise_acc, noise_loss)


def main():
    noise = [0.001, 0.005， 0.01, 0.05，0.1]
    dropout_list = [-1, 0.3, 0.5, 0.7, 0.9]
    rotations = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    color_list = ['r--', 'b--', 'c--', 'g--', 'k--']

    for (i, dropout) in enumerate(dropout_list):
        (rotation_loss, rotation_acc, noise_acc, noise_loss) = test_VGG_11(
            data_arg=True, dropout=dropout)
        plt.subplot(2, 2, 1)
        plt.plot(rotations, rotation_acc, color_list[i])
        plt.xlabel('rotations')
        plt.ylabel('rotation_acc')
        plt.title('Acc/rotations with different dropout rate')

        plt.subplot(2, 2, 2)
        plt.plot(rotations, rotation_loss, color_list[i])
        plt.xlabel('rotations')
        plt.ylabel('rotation_loss')
        plt.title('Loss/rotations with different dropout rate')

        plt.subplot(2, 2, 3)
        plt.plot(noise, noise_acc, color_list[i])
        plt.xlabel('noise')
        plt.ylabel('noise_acc')
        plt.title('Acc/noise with different dropout rate')

        plt.subplot(2, 2, 4)
        plt.plot(noise, noise_loss, color_list[i])
        plt.xlabel('noise')
        plt.ylabel('noise_loss')
        plt.title('Loss/noise with different dropout rate')

    plt.show()

main()
