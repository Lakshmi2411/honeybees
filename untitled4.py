
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

train_dataset="C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train"
labels = []
for class_ in os.listdir(train_dataset):
    labels.append(class_)
NUM_LABELS = len(labels)

print(labels)



# define the model architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes=4
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# load the data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/train',
    target_size=(224, 224),
    batch_size=32,classes=labels,
    class_mode='categorical')
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 12))
idx = 0

for i in range(3):
    for j in range(4):
        label = labels[np.argmax(train_generator[0][1][idx])]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
        ax[i, j].axis("off")
        idx += 1

# plt.tight_layout()
plt.suptitle("Sample Training Images", fontsize=21)
plt.show()

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/validate',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')







test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/Soundarya/Downloads/honeybees/bees-simple/bees/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# set up early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stop]
)

# plot training and validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
# Get the predicted labels and true labels for the test dataset
y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import confusion_matrix
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
# Calculate the confusion matrix
cm = confusion_matrix(test_generator.classes, y_pred)
report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)
# Plot the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax,cmap="cividis")