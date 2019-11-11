import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from keras.utils import to_categorical
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#DATADIR = "C:/Users/Malte Rossing/Desktop/VT1/Mini-project/PetImages/"
DATADIR = "C:/Users/Malte Rossing/Desktop/VT1/Mini-project/event_img/"
CATEGORIES = ["badminton", "bocce", "croquet", "polo", "RockClimbing", "rowing", "sailing", "snowboarding"]
#CATEGORIES = ["badminton", "bocce"]
#CATEGORIES = ["dog", "cat"]

for category in CATEGORIES:
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
		print(type(img))
		#img_array = np.array(img_array).reshape(img_array, (100, 100))
		plt.imshow(img_array.reshape(img_array.shape), cmap="gray")
		plt.show()
		break
	break
IMG_SIZE = 200

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		#class_num = to_categorical(CATEGORIES, num_classes = 8)
		for img in os.listdir(path):
			try:	
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				#array = np.array(img_array, dtype=float)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))	
				training_data.append([new_array, class_num])
			except Exception as e: 
				pass
create_training_data() 
print(len(training_data))

random.shuffle(training_data[:10])
for sample in training_data:
	print(sample[1])
X = []
Y = []
for features, label in training_data:
	X.append(features)
	Y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = to_categorical(Y, num_classes = 8)
pickle_out = open("XX.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close

pickle_out = open("YY.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close


"""
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
import os

dataset = "dataset.csv"
train = pd.read_csv(dataset)

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('event_img/badminton/img1'+'('+str(i+1)+').jpg', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

y = train['label'].values
y = to_categorical(y)
"""
