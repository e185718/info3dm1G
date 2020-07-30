import cv2, os, re,requests, time, bs4
from urllib.request import urlretrieve
from urllib import request as req
from urllib import error,parse
from PIL import Image
import numpy as np
import glob
import os.path

members = ["松本潤","二宮和也","相葉雅紀","大野智","櫻井翔"]
img_dir = "./arashi_image/"
cascade_file = "./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

def horizontal_flip(image, member):
    pil_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    pil_img.save(img_dir +member + "/face/"+ "A"+ str(num)+".jpg")
    
def vertical_flip(image, member):
    pil_img = image.transpose(Image.FLIP_TOP_BOTTOM)
    pil_img.save(img_dir +member + "/face/"+ "V"+ str(num)+".jpg")


def kaiten90(image, member):
	img_rotate = image.rotate(90)
	img_rotate.save(img_dir +member + "/face/" + "B"+ str(num)+".jpg")

def kaiten270(image, member):
    img_rotate = image.rotate(270)
    img_rotate.save(img_dir +member + "/face/" + "C"+ str(num)+".jpg")

def lighten(image, member):
    image = image + 15
    pil_img = Image.fromarray(image)
    pil_img.save(img_dir +member + "/face/" + "E"+ str(num)+".jpg")
    
def gray(image, member):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pil_img = Image.fromarray(image)
    pil_img.save(img_dir +member + "/face/" + "D"+ str(num)+".jpg")
    
def blur(image, member):
    image = cv2.GaussianBlur(image, (5, 5), 3)
    pil_img = Image.fromarray(image)
    pil_img.save(img_dir +member + "/face/" + "F"+ str(num)+".jpg")
    
def bigger(image, member):
	height, width = image.shape[:2]
	size=(int(width*1.5) , int(height*1.5))
	bigimg = cv2.resize(image,size)
	pil_img = Image.fromarray(bigimg)
	pil_img.save(img_dir +member + "/face/" + "G"+ str(num)+".jpg")
 
def smaller(image, member):
    height, width = image.shape[:2]
    size=(int(width*0.5) , int(height*0.5))
    bigimg = cv2.resize(image,size)
    pil_img = Image.fromarray(bigimg)
    pil_img.save(img_dir +member + "/face/" + "G"+ str(num)+".jpg")
 
 
nums = np.arange(300)
for member in members:
    print(member+"の処理をしています")
    for num in nums:
        img_path = img_dir + member + "/face/" + str(num) + ".jpg"
        if os.path.exists(img_path):
            im = Image.open(img_path)
            im2 = np.array(Image.open(img_path))
            im3 = cv2.imread(img_path)
            #horizontal_flip(im, member)
            vertical_flip(im, member)
            kaiten90(im, member)
            kaiten270(im, member)
            gray(im3, member)
            lighten(im2, member)
            #blur(im2, member)
            #bigger(im2, member)
            #smaller(im2, member)             
#for member in members:
#    pathes = img_dir + member + "/face"
 #   files = glob.glob(pathes +'/*')  
 #   for i, f in enumerate(files):
 #   	os.rename(f, os.path.join(pathes, '{}'.format(i)+ ".jpg"))