import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob

members = ["櫻井翔"]
img_dir = "./arasi_deta/downloads/"
cascade_file = "./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)
for member in members:
	files = glob.glob(img_dir+member+"/*.jpg")
	print("{}の写真は{}枚です。顔認識を始めます。".format(member, len(files)))
	error_count = 0
	for index, file in enumerate(files,1):
		print(file)
		img = cv2.imread(file)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cascade = cv2.CascadeClassifier(cascade_file)
		facerect = cascade.detectMultiScale(img_gray)
		print(len(facerect))
		if len(facerect) != 1:
			mistake_path = img_dir + member + "/mistake"
			if not os.path.exists(mistake_path):
				os.makedirs(mistake_path)
			cv2.imwrite(mistake_path  + "/" + str(index) +".jpg", img)
			error_count += 1
			continue
		for x,y,w,h in facerect:
			img = img[y:y+h, x:x+w]
		face_path = img_dir+member+"/face"
		if not os.path.exists(face_path):
			os.makedirs(face_path)
		cv2.imwrite(face_path  + "/" + str(index)+".jpg", img)
	print("{}のディレクトリにて、{}件中、{}件の写真で顔認識に失敗しました。".format(member, len(files), error_count))
