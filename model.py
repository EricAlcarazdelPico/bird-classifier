import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from generate_data import generate_data
import matplotlib.pyplot as plt
import pandas as pd

img_size = 224
batch_size = 128
epochs = 1

tf.keras.backend.clear_session()
tf.random.set_seed(42)

inputs = keras.Input(shape=(img_size, img_size, 3))

x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
#x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
# Get 225 from split_data
output = layers.Dense(225, activation='softmax')(x)

model = keras.Model(inputs, output, name='bird_classifier')

# from_logits=True
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(lr=0.1),
    metrics=["accuracy"]
)

train_data_generated, valid_data_generated, test_data_generated = generate_data(
    img_size, batch_size)

history = model.fit(train_data_generated, validation_data=valid_data_generated,
                    epochs=epochs, verbose=2)

model.save('myModel')
df = pd.DataFrame(history.history)

with open('myModel.csv', 'a') as f:
    df.to_csv(f, header=f.tell()==0)

# # Plot results
# acc = history.history['accuracy']
# # val_acc = history.history['val_acc']

# # loss = history.history['loss']
# # val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# # plt.subplot(1, 2, 2)
# # plt.plot(epochs_range, loss, label='Training Loss')
# # # plt.plot(epochs_range, val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss')
# plt.show()
