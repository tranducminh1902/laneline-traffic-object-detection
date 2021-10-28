import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import numpy as np

ts_model = tf.keras.models.load_model('weight_281021_2.h5')
ts_model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# img = image.load_img(r'C:\Users\Meobeos\Downloads\GTSRB\Test\00004.png', target_size=(150, 150))
# # print (img.shape)
# img_array  = image.img_to_array(img)
# print (img_array.shape)
# img_array  = np.expand_dims(img_array, axis=0)
# print (img_array.shape)
# prediction = ts_model.predict(img_array)

# print (prediction[0].argmax())