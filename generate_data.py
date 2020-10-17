from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from split_data import split_data
from plotImages import plotImages


def generate_data():

    def estructure_data(work_dir):
        # os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(work_dir, 'birds')
        if not os.path.exists(root):
            split_data()
        train_dir = os.path.join(root, 'train')
        valid_dir = os.path.join(root, 'valid')
        test_dir = os.path.join(root, 'test')
        return train_dir, valid_dir, test_dir

    def data_generated(work_dir, image_generator):
        return image_generator.flow_from_directory(
            directory=work_dir,
            target_size=(img_size, img_size),
            class_mode='binary',
            batch_size=batch_size)

    train_dir, valid_dir, test_dir = estructure_data(
        os.path.dirname(os.path.realpath(__file__)))
    
    img_size = 224
    batch_size = 128

    image_generator_train = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
    )
    train_data_generated = data_generated(train_dir, image_generator_train)

    image_generator = ImageDataGenerator(rescale=1./255)
    valid_data_generated = data_generated(valid_dir, image_generator)

    #image_generator_test = ImageDataGenerator(rescale=1./255)
    test_data_generated = data_generated(test_dir, image_generator)

    return train_data_generated, valid_data_generated, test_data_generated


def show_generated_data(train_data_generated):
    augmented_images = [train_data_generated[0][0][0] for i in range(5)]
    plotImages(augmented_images)

#inputs = keras.Input(shape=(img_size, img_size ,3))

#outputs = keras.layers.Dense(num_classes, activation='softmax')()
