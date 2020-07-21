import cv2
import numpy as np
import os
import sys
import glob
images=[]
labels=[]

IMG_WIDTH = 30
IMG_HEIGHT = 30
dim=(IMG_WIDTH ,IMG_HEIGHT)
for it in range(44):
	temp_img=[np.array(cv2.imread(file)) for file in glob.glob("gtsrb/"+str(it)+"/*.ppm")]
	temp_label=[it]*(len(temp_img))
	images=images+temp_img
	labels=labels+temp_label
images=[cv2.cvtColor(bla,cv2.COLOR_BGR2RGB) for bla in images]

images=[cv2.resize(np.array(img),dim) for img in images]


print(images,labels)

	
