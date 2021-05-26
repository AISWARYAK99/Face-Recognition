import cv2

from random import randrange

#load some pretrained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img = cv2.imread('me.jpg')

#must convert to gray scale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangles around the faces

#this is for a single face in a image
'''
(x, y, w, h) = face_coordinates[0]  #0 is given to get the first appearing face
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0), 10)  #255 corresponding to green in the format BGR , 10 is the thikness of the appearing reactangle
'''

#this is for a all faces appearing in a image
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 10)
    #randrange is for randomly displaying different colured rectangle

#displaying the image
cv2.imshow('Hey its me !!!!!',img)
cv2.waitKey()

#just to check
print('Code Completed')
