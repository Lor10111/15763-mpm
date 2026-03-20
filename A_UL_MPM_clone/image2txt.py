from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random

<<<<<<< HEAD
img_name = './examples/2D_bunny'
=======
img_name = './examples/small_bunny'
>>>>>>> TL_APIC_implicit
suffix = '.jpg'


image = img_to_array(load_img(img_name + suffix))
image = np.array(image, dtype=float)

rows = image.shape[0]
cols = image.shape[1]

file = open("2D_bunny.txt","w")
for i in range(rows):
    for j in range(cols):
        file.write(str(i)+" "+str(j)+" "+str(image[i][j][0])+" "+str(image[i][j][1])+" "+str(image[i][j][2])+"\n")
file.close()

print('done')