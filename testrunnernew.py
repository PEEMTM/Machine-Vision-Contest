from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

IMAGE_SIZE = (256,256)
BATCH_SIZE = 50

#Create model
input = Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3))
conv1 = Conv2D(64,3,activation='relu')(input)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D()(conv1)
conv2 = Conv2D(128,3,activation='relu')(pool1)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D()(conv2)
conv3 = Conv2D(256,3,activation='relu')(pool2)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D()(conv3)
conv4 = Conv2D(512,3,activation='relu')(pool3)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPool2D()(conv4)
conv5 = Conv2D(1024,3,activation='relu')(pool4)
conv5 = BatchNormalization()(conv5)
pool5 = MaxPool2D()(conv5)
flat = Flatten()(pool5)
dense1 = Dense(512,activation='sigmoid')(flat)
dense1 = Dropout(0.5)(dense1)
hidden = Dense(64, activation='relu')(dense1)
output = Dense(4, activation='softmax')(hidden)
model = Model(inputs=input, outputs=output)

model.compile(optimizer=Adam(lr = 1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Create generator (download dataset form https://drive.google.com/file/d/1Re4ededUgebu-vjVjHqjF9efC4xMfEih/view?usp=sharing)
datagen = ImageDataGenerator(rescale=1./255)

# Define the folder path, the filename list file path, and the output folder path
folder_path = "test images"
filename_list_path = "filelisttest.txt"
output_folder = "Test_Data_con"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the filename list file in read mode and read the lines
with open(filename_list_path, "r") as file:
    # Strip newline characters from each line and store in a list
    filenames = [line.strip() for line in file.readlines()]

global files
# Loop through each filename in the list
for filename in filenames:
    # Construct the full path of the file and the output path
    full_path = os.path.join(folder_path, filename)
    output_path = os.path.join(output_folder, filename)
    # Copy the file to the output folder
    shutil.copy(full_path, output_path)
        
file.close()

test_generator = datagen.flow_from_directory(
    'Test_Data_con',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode=None)

#Test Model
model = load_model('Food2.h5')

# Get the list of filenames in the directory
filenames = os.listdir('test images')

# Generate predictions for each image
predictions = []
for filename in filenames:
    # Load the image
    img = tf.keras.preprocessing.image.load_img(
        os.path.join('test images', filename),
        target_size=IMAGE_SIZE
    )
    # Convert the image to a numpy array
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    # Rescale the image
    img_arr /= 255.0
    # Add the image to the list of inputs
    predictions.append(img_arr)

# Convert the list of inputs to a numpy array
predictions = np.array(predictions)

# Generate predictions using the model
predict = model.predict(predictions)
print('confidence:\n', predict)

predict_class_idx = np.argmax(predict,axis = -1)
print('predicted class index:\n', predict_class_idx)

mapping = {('Burger', 0), ('Dimsum', 1), ('Ramen', 2), ('Sushi', 3)}
mapping = dict((v, k) for k, v in mapping)
predict_class_name = [mapping[x] for x in predict_class_idx]
# predict_class_idx = np.argmax(predict, axis=-1)[0]
# predict_class_name = mapping[predict_class_idx]
print('predicted class name:\n', predict_class_name)

class_sname = ''
with open('result.txt', 'w') as file:
    # Loop through each file in the folder
    for filename, predict_class in zip(filenames, predict_class_name):
        if predict_class == 'Burger' :
            class_sname = 'B'
        elif predict_class == 'Dimsum' :
            class_sname = 'D'
        elif predict_class == 'Ramen' :
            class_sname = 'R'
        elif predict_class == 'Sushi' :
            class_sname = 'S'
        # Write some text to the file
        file.write(filename+'::'+class_sname+'\n')
        
file.close()
