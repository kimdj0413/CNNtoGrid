import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class_names = ['apple', 'banana', 'coconut', 'grape',  'orange']

model = keras.models.load_model('fruitCNN86abcgo.keras')
# print(model.summary())
img_path = "C:/waytoMap/oranges.jpg"
img = tf.keras.utils.load_img(
    img_path, target_size=(180, 180)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

img = mpimg.imread(img_path)

# 이미지와 결과를 출력
plt.imshow(img)
plt.title(result)
plt.axis('off')  # 축을 숨김
plt.show()