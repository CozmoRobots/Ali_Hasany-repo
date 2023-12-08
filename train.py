import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import *

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

repertoire1 = 'train/'
repertoire2 = 'validation/'

image_length = 224
image_width = 224
batch_size = 32
image_size = (image_length, image_width)


train = image_dataset_from_directory(
    repertoire1,
    label_mode='categorical',
    image_size=image_size
)

test = image_dataset_from_directory(
    repertoire2,
    label_mode='categorical',
    image_size=image_size
)

data_augmentation = Sequential ([
    RandomFlip("horizontal", input_shape = (image_length, image_width, 3)),
    RandomRotation (0.3),
    RandomZoom (0.2)
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

#model = tf.keras.models.load_model("best_model.h5", compile=False)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)

# Define ModelCheckpoint callback to save the best model
model_checkpoint = ModelCheckpoint("best_model_4.h5", save_best_only=True)

history = model.fit(
    train,
    validation_data=test,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(epochs_range, loss, label='Training Loss', color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the figure
plt.savefig("training_validation_curves_aug.png")

plt.show()