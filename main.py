#Importing Libs
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy


## Step 1: Data Processing
#Defining the Input Shape
input_shape = (100, 100, 3)

#Defining File Paths
cwd = os.getcwd()
parent_directory = os.path.join(cwd, os.pardir) 
train_dir = os.path.abspath(os.path.join(parent_directory, 'AER850P2/Project 2 Data/Data/Train'))
val_dir = os.path.abspath(os.path.join(parent_directory, 'AER850P2/Project 2 Data/Data/Validation'))
test_dir= os.path.abspath(os.path.join(parent_directory, 'AER850P2/Project 2 Data/Data/Test'))
print('train dir:', train_dir)

#Data Augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,        
    shear_range=0.2,       
    zoom_range=0.2,        
    # horizontal_flip=True   # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'  
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

X_train, Y_train = train_generator.next()
##Step 2: Neural Network Architecture Design
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 20
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Number of batches per epoch
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size  # Number of batches for validation
)

##Step 4
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()