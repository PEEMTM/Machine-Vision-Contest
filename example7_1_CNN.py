#For Goole Colab Version
#https://colab.research.google.com/drive/1UwO27IYQVmsa-DD4sbxN7FdK_yfFAVkV?usp=share_link

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#Create model
input = Input(shape = (50,50,1))
conv1 = Conv2D(10,3,activation='relu',padding='same')(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(20,3,activation='relu',padding='same')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden = Dense(12, activation='relu')(flat)
output = Dense(4, activation='softmax')(hidden)
model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Read data from file (download form https://drive.google.com/file/d/1UACFvQ8QCFUQaBWz9umQxbUWLSQlh1co/view?usp=sharing)
N = 200
x_train = np.zeros((N,50,50,1),'float')
y_train = np.zeros((N),'float')
count = 0
# Define the folder path and the filename list file path
folder_path = "Test_Dataset"
# Loop through each subfolder in the root directory
for root, dirs, files in os.walk(folder_path):
    # Loop through each file in the subfolder
    for filename in files:
        class_name = root.split("\\").pop()
        im = cv2.imread("Dataset/"+class_name+"/"+filename,cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(50,50))
        x_train[count,:,:,0] = im/255.
        if class_name == "Burger":
            y_train[count] = 0
        elif class_name == "Dimsum":
            y_train[count] = 1
        elif class_name == "Ramen":
            y_train[count] = 2
        elif class_name == "Sushi":
            y_train[count] = 3
        else : raise Exception("false class")
            
        count += 1

y_train = to_categorical(y_train,num_classes=4)

#Train Model
h = model.fit(x_train, y_train, epochs=20)

plt.plot(h.history['accuracy'])

#Test Model
N = 15
x_test = np.zeros((N,50,50,1),'float')
y_test = np.zeros((N),'float')
count = 0
# Define the folder path and the filename list file path
test_folder_path = "testprint"
test_filename_list_path = "filetest.txt"
# Open the filename list file in read mode and read the lines
with open(test_filename_list_path, "r") as file:
    # Strip newline characters from each line and store in a list
    filenames = [line.strip() for line in file.readlines()]

file_names = []
# Loop through each subfolder in the root directory
for root, dirs, files in os.walk(test_folder_path):
    # Loop through each file in the subfolder
    for filename in files:
        # Check if the filename matches any of the filenames in the list
        if filename in filenames:
            file_names.append(filename)
            class_name = root.split("\\").pop()
            im = cv2.imread("Dataset/"+class_name+"/"+filename,cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im,(50,50))
            x_test[count,:,:,0] = im/255.
            if class_name == "Burger":
                y_train[count] = 0
            elif class_name == "Dimsum":
                y_train[count] = 1
            elif class_name == "Ramen":
                y_train[count] = 2
            elif class_name == "Sushi":
                y_train[count] = 3
            else : raise Exception("false class")
                
            count += 1

y_test = to_categorical(y_test,num_classes=4)

score = model.evaluate(x_test, y_test)
print('score (cross_entropy, accuracy):\n',score)

y_pred = model.predict(x_test)
print('confidence:\n', y_pred)
class_int = np.argmax(y_pred,axis = -1)+1
print('predicted class name:\n', class_int)

cm = confusion_matrix(np.argmax(y_test,axis = -1), np.argmax(y_pred,axis = -1))
print("Confusion Matrix:\n",cm)

count = 0
# Open a file in write mode
with open('result_a.txt', 'w') as file:
    for file_name in file_names :
        if class_int[count] == 1 :
            class_sname = 'B'
        elif class_int[count] == 2 :
            class_sname = 'D'
        elif class_int[count] == 3 :
            class_sname = 'R'
        elif class_int[count] == 4 :
            class_sname = 'S'
        # Write some text to the file
        file.write(str(file_name)+'::'+str(class_sname)+'\n')
        count += 1
        
    file.close()
        
plt.show()