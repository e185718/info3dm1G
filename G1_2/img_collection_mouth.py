import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob
#"二宮和也","櫻井翔","相葉雅紀","大野智","松本潤"
members = ["二宮和也","櫻井翔","相葉雅紀","大野智","松本潤"]
img_dir =  "./arashi_image/"
mouth_cascade_path = "./haarcascade_mcs_nose.xml"
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
for member in members:
	files = glob.glob(img_dir+member+"/*.jpg")
	print("{}の写真は{}枚です。顔認識を始めます。".format(member, len(files)))
	for index, file in enumerate(files,1):
		img = cv2.imread(file)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for i in range(1,20):
			minValue = i * 5
			mouth = mouth_cascade.detectMultiScale(img_gray, minSize=(minValue,minValue))
			if len(mouth) == 1:
				break
		if len(mouth) != 1:
				continue
		for x,y,w,h in mouth:
				img = img[y:y+h, x:x+w]
		mouth_path = img_dir+member+"/mouth"
		if not os.path.exists(mouth_path):
			os.makedirs(mouth_path)
		cv2.imwrite(mouth_path  + "/" + str(index)+".jpg", img)
	


