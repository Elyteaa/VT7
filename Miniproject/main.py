from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.constraints import max_norm
import os
import os.path
import imageio
from random import shuffle
from random import seed
from random import random
import shutil
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def label_img(name):
    word_label = name.split('_')[0]
    if word_label == 'badminton':
        return np.array([1, 0, 0, 0, 0, 0, 0, 0])
    elif word_label == 'bocce':
        return np.array([0, 1, 0, 0, 0, 0, 0, 0])
    elif word_label == 'croquet':
        return np.array([0, 0, 1, 0, 0, 0, 0, 0])
    elif word_label == 'polo':
        return np.array([0, 0, 0, 1, 0, 0, 0, 0])
    elif word_label == 'RockClimbing':
        return np.array([0, 0, 0, 0, 1, 0, 0, 0])
    elif word_label == 'Rowing':
        return np.array([0, 0, 0, 0, 0, 1, 0, 0])
    elif word_label == 'sailing':
        return np.array([0, 0, 0, 0, 0, 0, 1, 0])
    elif word_label == 'snowboarding':
        return np.array([0, 0, 0, 0, 0, 0, 0, 1])


def get_size_statistics(folder):
    heights = []
    widths = []
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        data = np.array(imageio.imread(path))  # PIL Image library
        heights.append(data.shape[0])
        widths.append(data.shape[1])
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))
    return (avg_height + avg_width) / 2


def load_training_data():
    train_data = []
    for img in os.listdir('./training'):
        label = label_img(img)
        path = os.path.join('./training', img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        train_data.append([np.array(img), label])

        # Basic Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        flip_img = flip_img.convert('L')
        flip_img = flip_img.resize((img_size, img_size), Image.ANTIALIAS)
        flip_img = np.array(flip_img)
        flip_img = np.fliplr(flip_img)
        train_data.append([flip_img, label])
        shuffle(train_data)
    return train_data


def load_crossval_data():
    crossval_data = []
    for img in os.listdir('./crossval'):
        label = label_img(img)
        path = os.path.join('./crossval', img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        crossval_data.append([np.array(img), label])

        # Basic Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        flip_img = flip_img.convert('L')
        flip_img = flip_img.resize((img_size, img_size), Image.ANTIALIAS)
        flip_img = np.array(flip_img)
        flip_img = np.fliplr(flip_img)
        crossval_data.append([flip_img, label])
        shuffle(crossval_data)
    return crossval_data


def load_testing_data():
    test_data = []
    for img in os.listdir('./test'):
        label = label_img(img)
        path = os.path.join('./test', img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        test_data.append([np.array(img), label])
        shuffle(test_data)
    return test_data


img_size = int(get_size_statistics('./event_img') / 6)

if os.path.exists('crossval'):
    shutil.rmtree('crossval')

if os.path.exists('training'):
    shutil.rmtree('training')

if os.path.exists('test'):
    shutil.rmtree('test')

os.mkdir('crossval')
os.mkdir('training')
os.mkdir('test')

print('directories created')

i = 0
seed(42)

for img in os.listdir('./event_img'):
    x = random()
    if (x < 0.64):
        newImgName = './training/' + img.split('_', 3)[2] + '_' + str(i) + '.jpg'
    elif(x<0.8):
        newImgName = './crossval/' + img.split('_', 3)[2] + '_' + str(i) + '.jpg'
    else:
        newImgName = './test/' + img.split('_', 3)[2] + '_' + str(i) + '.jpg'
    path = os.path.join('./event_img', img)
    image_data = np.array(imageio.imread(path))
    imageio.imwrite(newImgName, image_data)
    i += 1
print('images assigned')

train_data = load_training_data()
test_data = load_testing_data()
crossval_data = load_crossval_data()

trainImages = np.true_divide(np.array([i[0] for i in train_data]).reshape(-1, img_size, img_size, 1), 255.0)
trainLabels = np.array([i[1] for i in train_data])

testImages = np.true_divide(np.array([i[0] for i in test_data]).reshape(-1, img_size, img_size, 1), 255.0)
testLabels = np.array([i[1] for i in test_data])

crossValImages = np.true_divide(np.array([i[0] for i in crossval_data]).reshape(-1, img_size, img_size, 1), 255.0)
crossValLabels = np.array([i[1] for i in crossval_data])

print('data loaded')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1), padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, kernel_constraint=max_norm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=max_norm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(8, activation='softmax'))
print('finished building')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('model compiled')
print(model.summary())

model.fit(trainImages, trainLabels, validation_data=(crossValImages, crossValLabels), verbose=1, epochs=15)

print('model trained')

loss, accuracy = model.evaluate(testImages, testLabels, batch_size=1, verbose=1)
print('accuracy: ' + accuracy)
