# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:57:29 2019

@author: eloni
"""

import os
import csv
# =============================================================================
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import to_categorical
# from keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from tqdm import tqdm
# =============================================================================

image_path = 'c:/Users/eloni/Documents/__University/Semester 7/Machine Learning/Miniproject/event_dataset/event_img/'
image_categories = ['badminton', 'bocce', 'croquet', 'polo', 'RockClimbing', 'rowing', 'sailing', 'snowboarding']

#Read file names from each category, and place them in .csv files
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
   easy_training = round(easy * 0.8)
   medium_training = round(medium * 0.8)
   hard_training = round(hard * 0.8)
   
   temp_easy = 0
   temp_medium = 0
   temp_hard = 0
   
   for i in range(len(listofnames)):
      if listofnames[i].startswith('Easy') and temp_easy < easy_training:
         temp_easy += 1
         row = [listofnames[i], j]
         with open('training.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
         csvFile.close()
      elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
         temp_medium += 1
         row = [listofnames[i], j]
         with open('training.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
         csvFile.close()
      elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
         temp_hard += 1
         row = [listofnames[i], j]
         with open('training.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
         csvFile.close()
      else:
         with open('testing.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([listofnames[i]])
   
