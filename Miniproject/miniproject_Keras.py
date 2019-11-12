# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:57:29 2019

@author: eloni
"""

import os
import csv
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
# =============================================================================

image_path = 'c:/Users/eloni/Documents/__University/Miniproject/event_dataset/event_img/'
image_categories = ['badminton', 'bocce', 'croquet', 'polo', 'RockClimbing', 'Rowing', 'sailing', 'snowboarding']

#Read file names from each category, and place them in .csv files, create a header
with open('training.csv', 'a') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(['File name', 'Category', 'label'])
csvFile.close()

with open('testing.csv', 'a') as csvFile:
	writer = csv.writer(csvFile)
	writer.writerow(['File name', 'Category', 'label'])
csvFile.close()

for j in image_categories:
	listofnames = os.listdir(image_path + j)

	#Find the number of easy, mid and hard images
	easy = 0
	medium = 0
	hard = 0

	for i in range(len(listofnames)):
		if listofnames[i].startswith('Easy'):
			easy += 1
		elif listofnames[i].startswith('Medium'):
			medium += 1
		elif listofnames[i].startswith('Hard'):
			hard += 1

	#Separating 80% of the given data for a training set
	easy_training = round(easy * 0.9)
	medium_training = round(medium * 0.9)
	hard_training = round(hard * 0.9)

	temp_easy = 0
	temp_medium = 0
	temp_hard = 0

	for i in range(len(listofnames)):
		if listofnames[i].startswith('Easy') and temp_easy < easy_training:
			temp_easy += 1
			row = [listofnames[i], j, image_categories.index(j)]
			with open('training.csv', 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()
		elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
			temp_medium += 1
			row = [listofnames[i], j, image_categories.index(j)]
			with open('training.csv', 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()
		elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
			temp_hard += 1
			row = [listofnames[i], j, image_categories.index(j)]
			with open('training.csv', 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()
		elif listofnames[i] != 'Thumbs.db':
			with open('testing.csv', 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow([listofnames[i], j, image_categories.index(j)])
			csvFile.close()

train = pd.read_csv('training.csv')
train_image = []

#print(train.shape[0])
#print(train['File name'][0])

category_control = 0
#Iterate through all training data, loading the images
for i in tqdm(range(train.shape[0])):
	if image_categories[category_control] in train['File name'][i]:
		img = image.load_img(image_path + image_categories[category_control] + '/' + train['File name'][i], target_size=(28,28,1), grayscale=True)
		img = image.img_to_array(img)
		img = img/255
		train_image.append(img)
	elif category_control <= len(image_categories):
		category_control += 1
		img = image.load_img(image_path + image_categories[category_control] + '/' + train['File name'][i], target_size=(28,28,1), grayscale=True)
		img = image.img_to_array(img)
		img = img/255
		train_image.append(img)

X = np.array(train_image)

y = train['label'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test = pd.read_csv('testing.csv')
test_image = []

category_control = 0
#Iterate through all training data, loading the images
for i in tqdm(range(test.shape[0])):
	if image_categories[category_control] in test['File name'][i]:
		img = image.load_img(image_path + image_categories[category_control] + '/' + test['File name'][i], target_size=(28,28,1), grayscale=True)
		img = image.img_to_array(img)
		img = img/255
		test_image.append(img)
	elif category_control <= len(image_categories):
		category_control += 1
		img = image.load_img(image_path + image_categories[category_control] + '/' + test['File name'][i], target_size=(28,28,1), grayscale=True)
		img = image.img_to_array(img)
		img = img/255
		test_image.append(img)

testing = np.array(test_image)

# making predictions
prediction = model.predict_classes(testing)

correct = 0
for i in range(test.shape[0]):
	if test['label'][i] == prediction[i]:
		correct += 1

accuracy = correct / test.shape[0] * 100
print(accuracy)