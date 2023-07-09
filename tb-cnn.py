import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

train_dir = '/Users/maxkoustikov/Downloads/TB_Chest_Radiography_Database/train'
val_dir = '/Users/maxkoustikov/Downloads/TB_Chest_Radiography_Database/val'
test_dir = '/Users/maxkoustikov/Downloads/TB_Chest_Radiography_Database/test'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)  # Don't augment validation/test data

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(val_dir,
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

model = Sequential()

# Add the convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the tensor output by the convolutional layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=2,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print('test loss:', test_loss)
print('test acc:', test_acc)
