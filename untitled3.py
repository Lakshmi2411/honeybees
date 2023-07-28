# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 00:50:37 2023

@author: Soundarya
"""

import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


data_dir = 'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train'
class_names = os.listdir(data_dir)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    file_names = os.listdir(class_dir)
    num_files = len(file_names)
    print(f"Class {class_name} has {num_files} images")

batch_size=64
img_size=(150,150)
num_classes=4
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
train_generator = train_datagen.flow_from_directory('C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train',
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = train_datagen.flow_from_directory('C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/validate',
                                                         target_size=img_size,
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         subset='validation')
# Build the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1)
# Train the model
history = model.fit(train_generator,
                    epochs=35,
                    validation_data=validation_generator,callbacks=[early_stop])


test_generator = test_datagen.flow_from_directory(
    'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)
import os
import numpy as np
import matplotlib.pyplot as plt
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Generate predictions on the test set
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import confusion_matrix,classification_report
# Calculate the confusion matrix
cm = confusion_matrix(test_generator.classes, y_pred)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)
# Plot the confusion matrix
plt.imshow(cm)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(num_classes))
plt.yticks(np.arange(num_classes))
plt.show()
