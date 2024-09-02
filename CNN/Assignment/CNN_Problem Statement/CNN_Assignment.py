'''
The CIFAR-10 dataset is a popular benchmark dataset in the field of computer vision. It stands for the Canadian Institute for Advanced Research (CIFAR), which funded the collection of the dataset. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

Here are some key details about the CIFAR-10 dataset:

Classes: CIFAR-10 contains images across 10 different classes, with each class representing a specific object or category. The classes are:
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
Image Size: Each image in the CIFAR-10 dataset is of size 32x32 pixels. These are relatively small images compared to many other datasets, making CIFAR-10 suitable for quick experimentation and prototyping.
Color Channels: CIFAR-10 images are RGB (Red, Green, Blue) color images, meaning they have three color channels. Each pixel in the image is represented by three values, one for each color channel, ranging from 0 to 255.
Training and Test Split: The dataset is divided into training and test sets. The training set consists of 50,000 images, while the test set contains 10,000 images. This split ensures that models are trained on one set of data and evaluated on another, unseen set to assess their generalization performance.
Challenges: CIFAR-10 poses several challenges to machine learning models due to its relatively small image size, low resolution, and the presence of multiple classes. Models trained on CIFAR-10 must learn to distinguish between various objects with limited visual information, making it a challenging dataset for tasks such as image classification.
Overall, CIFAR-10 serves as a standard benchmark dataset for evaluating the performance of machine learning algorithms, particularly in tasks related to image classification and object recognition. Many research papers and studies in the field of computer vision use CIFAR-10 as a baseline dataset for comparison and evaluation of new techniques and algorithms.
'''

import tensorflow as tf


tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)


import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Design the Network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Adding dropout layer for regularization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# Num of Parameters
# [(w*h*d)+1]*k 
# w = width; h = height; d = filters from previous layer
# k = current layer filters

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# Load the cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# reshaping test_images to 2d 
arr_reshaped = test_images.reshape(test_images.shape[0], -1)

# converting reshaped array to dataframe
df = pd.DataFrame(arr_reshaped)  

# selecting 100 rows randomly
df1 = df.sample(n = 100)

# Saving small amount of data for testing
df1.to_csv('cifar10_cnn.csv', index=False)


train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_val, y_val))


# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# test_acc


model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.weights.h5")




#######################
# Testing on new data

from tensorflow.keras.models import model_from_json

import pandas as pd

# from keras import model_from_json 

# opening and store file in a variable

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.weights.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
# loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test1 = pd.read_csv("cifar10_cnn.csv")


arr_img = test1.to_numpy()

test_pred = arr_img.reshape((len(arr_img), 32, 32, 3))

test_pred = test_pred.astype('float32') / 255

predictions = pd.DataFrame(loaded_model.predict(test_pred))

