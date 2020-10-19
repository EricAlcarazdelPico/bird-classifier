import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from generate_data import *

img_size = 224
batch_size = 128


inputs = keras.Input(shape=(img_size, img_size, 3))

x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Droput(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
# Get 225 from split_data
output = layers.Dense(225, activation='softmax')(x)

model = keras.Model(inputs, output, name='bird_classifier')

train_data_generated, valid_data_generated, test_data_generated = generate_data(
    img_size, batch_size)
