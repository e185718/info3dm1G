import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
import numpy as np
import time
from tensorflow.keras.utils import to_categorical
members = ["松本潤","二宮和也","相葉雅紀"]
nb_members = len(members)
img_width, img_height = 70, 70

input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_members, activation='softmax'))

vgg_model = Model(inputs=vgg16.inputs, outputs=top_model(vgg16.outputs))

from tensorflow.keras.models import load_model
vgg_model.load_weights('./results/Weight.h5')

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def img_predict(filename):
	img = image.load_img(filename, target_size=(img_height, img_width))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    # これを忘れると結果がおかしくなるので注意
	x = x / 255.0
    #表示
	plt.imshow(img)
	plt.show()
    # 指数表記を禁止にする
	np.set_printoptions(suppress=True)
    #画像の人物を予測    
	pred = vgg_model.predict(x)[0]
    #結果を表示する
	print("'松本潤': 0, '二宮和也': 1, '相葉雅紀': 2")
	print(pred*100)
    
    
import glob
#テスト用の画像が入っているディレクトリのpathを()に入れてください
test = glob.glob('./test_sample_original/*')
#数字は各自入力
img_predict(test[0])