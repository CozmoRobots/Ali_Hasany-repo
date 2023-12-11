import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

# ########################### MODEL TRAINING SCRIPT ########################################
# This should work with any python environment with python 3 and and tensorflow > 2.0.0
# but bear in mind that I used the following environment to run this script
# Requirements:
# tensorflow 2.13.1
# python 3.8.10
# OS: Ubuntu 20.04.6 LTS
# numpy 1.24.3
# matplotlib 3.7.4
#############################################################################################



# repertoire1 is the path to our card data set
repertoire1 = 'train/'

# Defining the image size
image_length = 224
image_width = 224
batch_size = 32
image_size = (image_length, image_width)

seed_value = 42

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    repertoire1,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed_value
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    repertoire1,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed_value 
)

data_augmentation = Sequential([
    RandomFlip("horizontal", input_shape=(image_length, image_width, 3)),
    RandomRotation(0.3),
    RandomZoom(0.2)
])

model = Sequential()
model.add(data_augmentation)
model.add(InputLayer(input_shape=(image_length, image_width, 3)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(53, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

# Defining Early Stopping callback to store best weights
early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)

# Defining ModelCheckpoint callback to save the best model
model_checkpoint = ModelCheckpoint("best_model_4.h5", save_best_only=True)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# printing the training and validation accuracy AND training and validation loss graphs

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("training_validation_curves_aug.png")

plt.show()