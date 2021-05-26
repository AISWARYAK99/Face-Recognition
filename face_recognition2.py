import cv2

from random import randrange

#load some pretrained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#to capture video from webcam
webcam = cv2.VideoCapture(0)  #0 indicating default webcam,if we provide name of video file in it it will take it as a video

#iterate forever over frames
while True:
    #read the current frame
    successfull_frame_read, frame = webcam.read()#successful_frame_read just returns boolean true/false

    #must convert to grayscale
     
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)
    #randrange is for randomly displaying different colured rectangle

    cv2.imshow('Hey its me !!!!',frame)
    key=cv2.waitKey(1)

    #stop if Q is pressed
    if key==81 or key==113:
        break


#releasing webcam
webcam.release()



#just to check
print('Code Completed')