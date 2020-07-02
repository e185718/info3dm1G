import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob

members = ["松本潤","二宮和也","相葉雅紀",]
img_dir = "./arasi_deta/downloads/"
face_cascade_path = "./haarcascade_frontalface_alt.xml"
eye_cascade_path = "./haarcascade_righteye_2splits.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
for member in members:
	files = glob.glob(img_dir+member+"/*.jpg")
	print("{}の写真は{}枚です。顔認識を始めます。".format(member, len(files)))
	error_count = 0
	for index, file in enumerate(files,1):
		img = cv2.imread(file)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for i in range(1,20):
			minValue = i * 5
			eyes = eye_cascade.detectMultiScale(img_gray, minSize=(minValue,minValue))
			if len(eyes) == 1:
				break
		if len(eyes) != 1:
			continue
		for x,y,w,h in eyes:
			img = img[y:y+h, x:x+w]	
		face_path = img_dir+member+"/face3"
		if not os.path.exists(face_path):
			os.makedirs(face_path)
		cv2.imwrite(face_path  + "/" + str(index)+".jpg", img)
	print("{}のディレクトリにて、{}件中、{}件の写真で顔認識に失敗しました。".format(member, len(files), error_count))
