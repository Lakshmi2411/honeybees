# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 01:50:22 2023

@author: Soundarya
"""

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import os
import cv2

# Define the path to the train, test, and validation folders
train_dir = 'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train'
test_dir = 'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/test'
val_dir = 'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/validate'

# Preprocess the image data
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img

# Load the image data and preprocess it
train_images = []
train_labels = []

for label, class_name in enumerate(os.listdir("C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train")):
    class_path = os.path.join("train", class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224)) # Resize image to 224x224
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            test_images.append(preprocess_image(image_path))
            test_labels.append(int(os.path.basename(root)) - 1)

val_images = []
val_labels = []
for root, dirs, files in os.walk(val_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            val_images.append(preprocess_image(image_path))
            val_labels.append(int(os.path.basename(root)) - 1)

# Train an SVM classifier on the preprocessed image data
svm_model = svm.SVC(kernel='linear')
svm_model.fit(np.array(train_images).reshape(len(train_images), -1), np.array(train_labels))

# Evaluate the SVM model on the test data
y_pred = svm_model.predict(np.array(test_images).reshape(len(test_images), -1))
accuracy = accuracy_score(test_labels, y_pred)
print('Test accuracy:', accuracy)

# Print classification report and confusion matrix
target_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
print('Classification Report:')
print(classification_report(test_labels, y_pred, target_names=target_names))
print('Confusion Matrix:')
print(confusion_matrix(test_labels, y_pred))
