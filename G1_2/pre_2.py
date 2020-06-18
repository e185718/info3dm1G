import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
import time

members = [""]
nb_members = len(members)
img_width, img_height = 70, 70

# トレーニング用とバリデーション用の画像格納先
train_data_dir = './FaceEdited'
validation_data_dir = './test'
#トレーニングデータ用の画像数
nb_train_samples = 120
#バリデーション用の画像数
nb_validation_samples = 30
#バッチサイズ
batch_size = 30
#エポック数
nb_epoch = 10

train_datagen = ImageDataGenerator(
  rescale=1.0 / 255,
  #すでに画像の水増し済みの方は、下記２行は必要ありません。
  #zoom_range=0.2,
  #horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=members,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=members,
  class_mode='categorical',
  batch_size=batch_size,
  shuffle=True)

