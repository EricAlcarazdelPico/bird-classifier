from split_data import split_data
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '/birds')
if not os.path.exists(root):
    print(os.path.dirname(os.path.realpath(__file__)))
    split_data()

train_dir = os.path.join(root, 'train')
valid_dir = os.path.join(root, 'valid')
test_dir = os.path.join(root, 'test')

img_size = 224
batch_size = 128

# Get this function from another file


def plotImages(images_arr):
    fig_size = 4 * len(images_arr)
    fig, axes = plt.subplots(1, len(images_arr), figsize=(fig_size, fig_size))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


image_generator_train = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

train_data_generated = image_generator_train.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    class_mode='binary',
    batch_size=batch_size)

augmented_images = [train_data_generated[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_generator_valid = ImageDataGenerator(rescale=1./255)
valid_data_generated = image_generator_valid.flow_from_directory(
    directory=valid_dir, 
    target_size=(img_size, img_size), 
    class_mode='binary', 
    batch_size=batch_size)

image_generator_test = ImageDataGenerator(rescale=1./255)
test_data_generated = image_generator_test.flow_from_directory(
    directory=test_dir, 
    target_size=(img_size, img_size), 
    class_mode='binary', 
    batch_size=batch_size)

#inputs = keras.Input(shape=(img_size, img_size ,3))

#outputs = keras.layers.Dense(num_classes, activation='softmax')()
