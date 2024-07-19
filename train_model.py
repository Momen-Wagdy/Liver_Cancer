import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Path to the directory where photos are saved in class-based folders
destination_dir = r'D:\Projects\Omnia\Ultrasound\liver_ultrasound.v11i.tensorflow\Classified_data'

# Function to create the model
def create_model():
    base_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model():
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        destination_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        destination_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=[early_stopping, model_checkpoint]
    )

if __name__ == "__main__":
    train_model()
