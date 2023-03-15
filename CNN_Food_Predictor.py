from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'Dataset',
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode = 'rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    'Validationset',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'Test_Dataset',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')



#Train Model
checkpoint = ModelCheckpoint('Food2.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit_generator(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])



#Test Model
model = load_model('Food2.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)


test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('confidence:\n', predict)

predict_class_idx = np.argmax(predict,axis = -1)
print('predicted class index:\n', predict_class_idx)

mapping = dict((v,k) for k,v in test_generator.class_indices.items())
predict_class_name = [mapping[x] for x in predict_class_idx]
print('predicted class name:\n', predict_class_name)

cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
print("Confusion Matrix:\n",cm)

plt.show()