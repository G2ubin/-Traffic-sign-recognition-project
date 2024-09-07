import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# Задаем параметры
train_data_dir = '/home/aladin/proj_dop/train_dataset'
test_data_dir = '/home/aladin/proj_dop/test_dataset'
img_width, img_height = 150, 150
batch_size = 15
epochs = 50  # Увеличим количество эпох

# Добавляем аугментацию данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

inputs = Input(shape=(img_width, img_height, 3))

x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)  # Добавляем слой Dropout
x = Dense(16, activation='relu')(x)
# Выходной слой
outputs = Dense(1, activation='sigmoid')(x)

# Создаем модель
model = Model(inputs, outputs)

# Компилируем модель с настройками оптимизатора Adam и измененным learning rate
opt = Adam(learning_rate=0.0001)  # Уменьшаем learning rate
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

checkpoint_path = 'best_model.keras'

# Создаем callback для сохранения модели с лучшей точностью
checkpoint = ModelCheckpoint(checkpoint_path, 
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1
                            )

# Обучаем модель
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[checkpoint]
)

best_model = load_model(checkpoint_path)

# Оцениваем лучшую модель на тестовом наборе данных
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)
print("Best Test Loss:", test_loss)
print("Best Test Accuracy:", test_accuracy)
