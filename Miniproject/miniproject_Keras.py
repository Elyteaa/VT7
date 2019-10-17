# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:57:29 2019

@author: eloni
"""

import os
import csv

image_path = 'c:/Users/eloni/Documents/__University/Semester 7/Machine Learning/Miniproject/event_dataset/event_img/'

#Read file names from the first data folder
listofnames = os.listdir(image_path + 'badminton')
#print(listofnames[0])

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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'badminton']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'badminton']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'badminton']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'bocce')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'bocce']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'bocce']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'bocce']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'croquet')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'croquet']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'croquet']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'croquet']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'polo')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'polo']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'polo']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'polo']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'RockClimbing')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'RockClimbing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'RockClimbing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'RockClimbing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'rowing')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'rowing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'rowing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'rowing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'sailing')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'sailing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'sailing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'sailing']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])

listofnames = os.listdir(image_path + 'snowboarding')
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
   
#ans = easy + medium + hard
#print(ans)
      
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
      row = [listofnames[i], 'snowboarding']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Medium') and temp_medium < medium_training:
      temp_medium += 1
      row = [listofnames[i], 'snowboarding']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   elif listofnames[i].startswith('Hard') and temp_hard < hard_training:
      temp_hard += 1
      row = [listofnames[i], 'snowboarding']
      with open('training.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()
   else:
      with open('testing.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow([listofnames[i]])