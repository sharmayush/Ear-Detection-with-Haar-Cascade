import skimage.io as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


face_cascade = cv2.CascadeClassifier('/home/lenovo/Documents/project1/haarcascade_frontalface_default.xml')
left_ear_cascade = cv2.CascadeClassifier('/home/lenovo/Documents/project1/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('/home/lenovo/Documents/project1/haarcascade_mcs_rightear.xml')

img = cv2.imread("/home/lenovo/Documents/project1/img4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imageio.imshow(gray)

faces= face_cascade.detectMultiScale(gray, 1.3, 5)
# print(faces.shape)

faces_array=[]
for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
    faces_array.append(gray[y:y+h, x:x+w])
    imageio.imshow(faces_array[0])
    # print(faces)


Lears= left_ear_cascade.detectMultiScale(gray, 1.3, 5)

print(Lears)

left_array=[]
for (x, y, w, h) in Lears:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    left_array.append(gray[y:y+h, x:x+w])
    imageio.imshow(left_array[0])
    # print(Lears)

Rears= right_ear_cascade.detectMultiScale(gray, 1.3, 5)

right_array=[]
for (x, y, w, h) in Rears:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 3)
    right_array.append(gray[y:y+h, x:x+w])
    imageio.imshow(right_array[0])
    # print(Rears)

# print(Rears)

gozler=[]
for (x,y,w,h) in Lears:
    gozler.append(gray[y:y+h, x:x+w])
imageio.imshow(gozler[0])
for gz in gozler:
   plt.imshow(gz)
   plt.show()   
   pd.DataFrame({'Lears':str(gozler[0])},index=[0]).to_csv('Lefteardata.csv')

print(gozler)

gozler1=[]
for (x,y,w,h) in Rears:
    gozler1.append(gray[y:y+h, x:x+w])  
imageio.imshow(gozler1[0])
for gz in gozler1:
   plt.imshow(gz)
   plt.show()   
   pd.DataFrame({'Rears':str(gozler1[0])},index=[0]).to_csv('Righteardata.csv')

print(gozler1)





