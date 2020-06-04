import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob

members = ["二宮和也"]
img_dir = "./arasi_deta/downloads/"
cascade_file = "./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)
resized_image_list = []
count_ = []

for member in members:
    pathes = img_dir + member + "/face"
    files = glob.glob(pathes +'/*')  
    for i, f in enumerate(files):
        print(i)
        os.rename(f, os.path.join(pathes, '{}'.format(i)+ ".jpg"))
        
for member in members:
    files = os.listdir(img_dir+member+"/face")
    count = 0
    for file in files:
        if re.search(".jpg", file):
            count += 1
    resized_image= []
    nums = np.arange(count)
    error_count = 0
    for num in nums:
        try:            
            path = img_dir+member+"/face/"+str(num)+".jpg"
            img = cv2.imread(path)
            img = cv2.resize(img, (70,70))
            resized_image.append(img)
        except:
            error_count += 1
            continue
    resized_image_list.append(resized_image)
    print(member+"の写真を"+str(count)+"枚ダウンロードが終了しました。エラーは"+str(error_count)+"件でした")

from tensorflow.keras.utils import to_categorical
 
for i, photo in enumerate(resized_image_list):
    if i ==0:
        X = np.array(photo)
        y = np.array([0]*len(photo))
    else:
        X_ = np.array(photo)
        y_ = np.array([i] * len(photo))
        X = np.concatenate([X, X_], axis = 0)
        y = np.concatenate([y, y_], axis = 0)
        
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]
print("総データ数：",len(X))
 
X_train = X[:int(len(X)*0.7)]
y_train = y[:int(len(X)*0.7)]
X_test = X[int(len(X)*0.7):]
y_test = y[int(len(X)*0.7):]
 
y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)
 
print("学習データ数：",len(X_train))
print("テストデータ数：",len(X_test))

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping
 
input_tensor = Input(shape=(70, 70, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
 
model = Sequential()
model.add(Flatten(input_shape=vgg16.output_shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
 
model = Model(inputs=vgg16.inputs, outputs=model(vgg16.outputs))
 
for layer in model.layers[:15]:
    layer.trainable=False
 
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.00001, momentum=0.9), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer = optimizers.RMSprop(lr=1e-5), metrics=['accuracy'])
 
early_stopping = EarlyStopping(patience=5)
history = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(X_test, y_test),callbacks=[early_stopping])
score = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))
 
model.save_weights("./h5_person.hdf5")
 
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

def character(img):
    #予測したい画像データ(img)をopenCVを用いて(70,70)にリサイズ
    img = cv2.resize(img, (70, 70))
    #b,g,r = cv2.split(img) 
    #img = cv2.merge([r,g,b])
    pred = np.argmax(model.predict(np.array([img]), verbose=0))
    #plt.imshow(img)
    #plt.show
    print(pred)
    return ["二宮和也"][pred]
    
nums = np.arange(1,2)
count = 0
for num in nums:
    path = img_dir + "test/" + str(num) + ".jpg"
    img = cv2.imread(path)
    answer = ["二宮和也"][(num-1)//1]
    
    if character(img) == answer:
        count += 1
        print("アタリ", character(img), answer)
    else:
        print("ハズレ",character(img), answer)
        
print("正答率は{}/25でした".format(count))