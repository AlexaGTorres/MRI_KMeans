import numpy as np
import cv2 as cv
from PIL import Image #needed for rgb extraction
import os #needed for iterating through folder
from os import listdir #needed for iterating through folder
import pandas as pd #needed for going through .csv file

import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans 

#conversion
# img_array = np.load('MRNet-v1.0/valid/coronal/1249.npy')
# img = Image.fromarray(img_array, 'RGB')
# img.save('Converts/Coronal/Meniscus/1249.png')
# #img.show('Converts/Coronal/Meniscus/1221.png')


# #Sagittal Acl
# #read images
# imgA1 = cv.imread('Converts/Sagittal/Acl/1175.png')
# imgA2 = cv.imread('Converts/Sagittal/Acl/1179.png')
# imgA3 = cv.imread('Converts/Sagittal/Acl/1181.png')
# imgA4 = cv.imread('Converts/Sagittal/Acl/1182.png')
# imgA5 = cv.imread('Converts/Sagittal/Acl/1183.png')
# imgA6 = cv.imread('Converts/Sagittal/Acl/1185.png')
# imgA7 = cv.imread('Converts/Sagittal/Acl/1187.png')
# imgA8 = cv.imread('Converts/Sagittal/Acl/1219.png')
# imgA9 = cv.imread('Converts/Sagittal/Acl/1193.png')
# imgA10 = cv.imread('Converts/Sagittal/Acl/1198.png')

# #concatanate vertically
# vertiSA = np.concatenate((imgA1, imgA2, imgA3, imgA4, imgA5, imgA6, imgA7, imgA8, imgA9, imgA10), axis = 0)

# #window size
# imSA = cv.resize(vertiSA, (1000, 800))

# #open
# cv.imshow('Acl', imSA)
# cv.waitKey(0)
# cv.destroyAllWindows()

# #Coronal Mensicus
# #read images
# imgM1 = cv.imread('Converts/Coronal/Meniscus/1221.png')
# imgM2 = cv.imread('Converts/Coronal/Meniscus/1222.png')
# imgM3 = cv.imread('Converts/Coronal/Meniscus/1223.png')
# imgM4 = cv.imread('Converts/Coronal/Meniscus/1226.png')
# imgM5 = cv.imread('Converts/Coronal/Meniscus/1227.png')
# imgM6 = cv.imread('Converts/Coronal/Meniscus/1247.png')
# imgM7 = cv.imread('Converts/Coronal/Meniscus/1231.png')
# imgM8 = cv.imread('Converts/Coronal/Meniscus/1232.png')
# imgM9 = cv.imread('Converts/Coronal/Meniscus/1233.png')
# imgM10 = cv.imread('Converts/Coronal/Meniscus/1249.png')

# #concatanate vertically
# vertiSM = np.concatenate((imgM1, imgM2, imgM3, imgM4, imgM5, imgM6, imgM7, imgM8, imgM9, imgM10), axis = 0)

# #window size
# imSM = cv.resize(vertiSM, (1000, 800))

# #open
# cv.imshow('Meniscus', imSM)
# cv.waitKey(0)
# cv.destroyAllWindows()


# #extracting RGB values from a singular image
# img = Image.open('Converts/Coronal/Meniscus/1221.png')
# img.convert('RGB')

# r_total = 0
# g_total = 0
# b_total = 0

# width, height = img.size

# for x in range(0, width):
#     for y in range(0, height):
#         r, g, b = img.getpixel((x,y))
#         r_total += r
#         g_total += g
#         b_total += b
# print(r_total, g_total, b_total)


# #extract RGB values and save them in lists
# train_list = []
# val_list = []

# fol = "Converts/Coronal/Meniscus"

# #interate through folder
# for images in os.listdir(fol):
#     #check if files in folder end in 'png'
#     if(images.endswith('.png')):
#         #print(images)
#         img = Image.open('Converts/Coronal/Meniscus/' + images)
#         img.convert('RGB')
#         r_total = 0
#         g_total = 0
#         b_total = 0

#         width, height = img.size

#         for x in range(0, width):
#             for y in range(0, height):
#                 r, g, b = img.getpixel((x,y))
#                 r_total += r
#                 g_total += g
#                 b_total += b
#         #print(r_total, g_total, b_total)
#         train_list += [r_total, g_total, b_total]
#         print(train_list)


#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------


#Iterate through Training folder to convert npy to png
# for images in os.listdir('Dataset/Axial/Not Converted/Training'):
#     if (images.endswith('.npy')):
#         #Convert to png
#         img_array = np.load('Dataset/Axial/Not Converted/Training/' + images)
#         img = Image.fromarray(img_array, 'RGB')
#         img.save('Dataset/Axial/Converted/Training/' + images + '.png')

# #Iterate through Validation folder to convert npy to png
# for images in os.listdir('Dataset/Axial/Not Converted/Validation'):
#     if (images.endswith('.npy')):
#         #Convert to png
#         img_array = np.load('Dataset/Axial/Not Converted/Validation/' + images)
#         img = Image.fromarray(img_array, 'RGB')
#         img.save('Dataset/Axial/Converted/Validation/' + images + '.png')
    
# var = os.listdir('Dataset/Axial/Not Converted/Validation')
# print(var)

#Extract RGB values and save them in lists
train_list = []
val_list = []

folTrain = "Dataset/Axial/Converted/Training"
folValid = "Dataset/Axial/Converted/Validation"

var = os.listdir(folTrain)
#print(var)

# #Iterate through training folder
# for images in os.listdir(folTrain):
#     #Check if files in folder end in 'png'
#     img = cv.imread('Dataset/Axial/Converted/Training/' + images)
# print(len(img))

#print('\nTraining list RGB values\n')

# kmeans = KMeans(n_clusters = 2) #number of clusters to form as well as number of centroids to generate

# fitThing = kmeans.fit(img)#Compute kmeans clustering

# labels = kmeans.labels_ #labels of each point
# centers = kmeans.cluster_centers_ #cluster centers

# #x_data = [i for i in range(img.shape[1])]
# #print(len(centers))

# #print(centers.shape[0])
# #print(centers.shape[1])

# plt.scatter(img.shape[0], kmeans.cluster_centers_.shape[0], color = 'red',alpha=0.2,s=70)
# plt.scatter(img.shape[0], kmeans.cluster_centers_.shape[1], color = 'blue',alpha=0.2,s=50)
#plt.show()




















#dont need to run .predict on training data only validation
#cluster centers are what youd use to graph the plots
    
# #Iterate through validation folder
# for images in os.listdir(folValid):
#     #Check if files in folder end in 'png'
#     if(images.endswith('.png')):
#         img = Image.open('Dataset/Axial/Converted/Validation/' + images)
#         img.convert('RGB')
#         img_converted_to_np_type = np.array(img)
#         #print(type(img_converted_to_np_type))

#         val_list.append(img_converted_to_np_type)
        
        
        # r_total = 0
        # g_total = 0
        # b_total = 0

        # width, height = img.size

        # for x in range(0, width):
        #     for y in range(0, height):
        #         r, g, b = img.getpixel((x,y))
        #         r_total += r
        #         g_total += g
        #         b_total += b
        # rgbsV = [str(r_total) + ' ' + str(g_total) + ' ' + str(b_total)]
        # val_list += [rgbsV]

# print('\nValidation list RGB values\n')
# print(val_list)

for images in os.listdir(folTrain):
    #Check if files in folder end in 'png'
    if(images.endswith('.png')):
        imgT = Image.open('Dataset/Axial/Converted/Training/' + images)
        imgT.convert('RGB')

        imgT = imgT.resize((44,44))

        imgT_converted_to_np_type = np.array(imgT)
        #print(type(img_converted_to_np_type))

        train_list.append(imgT_converted_to_np_type)


#Read .csv file with pandas dataframe
#Start with training
#print('\nTraining labels\n')
dfT = pd.read_csv('MRNet-v1.0/train-acl.csv', header = None, nrows = 400)
trainLabel = dfT.values.tolist()
trainLA = np.array(trainLabel)
ah = trainLA[:,1:]

#print(ah)

ordered_training_data = []
ordered_training_labels = []

#train_images_class_1 = []
#train_images_class_0 = []
count_ones = 0
count_zeros = 0

for x in range(len(ah)):
    if ah[x] == 1:
        #print("Label is 1")
        ordered_training_data.append(train_list[x])
        ordered_training_labels.append(ah[x])
        count_ones += 1

for x in range(len(ah)):
    if ah[x] == 0:
        ordered_training_data.append(train_list[x])
        ordered_training_labels.append(ah[x])
        count_zeros += 1

print(ordered_training_labels)
#print(train_list[0].shape)

ordered_training_data = np.array(ordered_training_data)
#print(ordered_training_data.shape)
ordered_training_data = ordered_training_data.reshape(len(ordered_training_data), -1)
#print(ordered_training_data.shape)

kmeans = KMeans(n_clusters = 2) #number of clusters to form as well as number of centroids to generate

fitThing = kmeans.fit(ordered_training_data)#Compute kmeans clustering

#print(fitThing.cluster_centers_.shape)

x_data = [i for i in range(fitThing.cluster_centers_.shape[1])]
#print(len(centers))

#print(centers.shape[0])
#print(centers.shape[1])

plt.scatter(x_data, fitThing.cluster_centers_[0], color = 'red',alpha=0.2,s=70)
plt.scatter(x_data, fitThing.cluster_centers_[1], color = 'blue',alpha=0.2,s=50)
plt.show()

#print(fitThing.labels_)

total_ones = 0
total_zeros = 0

# for i in range(85):
#     if fitThing.labels_[i] == 1:
#         total_ones += 1
#     elif fitThing.labels_[i] == 0:
#         total_zeros += 1

# print(total_ones, total_zeros)

#print(ordered_training_labels)
#print("Total images without an ACL tear: ", count_zeros)
#print("Total images with an ACL tear: ", count_ones)

#print(len(trainLabel))

#print(train_images_class_1)
#print(train_images_class_0)

#trainTotal = train_images_class_1 + train_images_class_0

#print(trainTotal)

# Now validation
#print('\nValidation labels\n')
# dfV = pd.read_csv('MrNet-v1.0/valid-acl.csv', header = None, nrows = 100)
# validLabel = dfV.values.tolist()
# validLA = np.array(validLabel)
# vah = validLA[:,1:]

# valid_images_class_1 = []
# valid_images_class_0 = []

# for x in vah:
#     if 1 in vah:
#         valid_images_class_1.append(1)
#     if 0 in vah:
#         valid_images_class_0.append(0)

# #print(valid_images_class_1)
# #print(valid_images_class_0)

# validTotal = valid_images_class_1 + valid_images_class_0

#print(validTotal)
