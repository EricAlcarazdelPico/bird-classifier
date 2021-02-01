import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from generate_data import generate_data
import shutil
import pandas as pd

model = keras.models.load_model('myModel')

tf.keras.backend.clear_session()
tf.random.set_seed(42)

img_size = 224
batch_size = 128
epochs = 2

train_data_generated, valid_data_generated, test_data_generated = generate_data(
    img_size, batch_size)

history = model.fit(train_data_generated, validation_data=valid_data_generated,
                    epochs=epochs, verbose=2)

shutil.rmtree('myModel')
model.save('myModel')
df = pd.DataFrame(history.history)

with open('myModel.csv', 'a') as f:
    df.to_csv(f, header=f.tell()==0)

print(model)