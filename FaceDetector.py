from cv2 import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#image file
#img = cv2.imread('rdj1.jpg')
#img = cv2.imread('rdj2.jpg')
#img = cv2.imread('rdj3.jpg')
webcam = cv2.VideoCapture('testvideo.mp4')

#iterate over frames
while True:
    successful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinate = trained_face_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)

    #display image
    cv2.imshow('My Face Dector', frame)
    key = cv2.waitKey(1) 

    ##stop if q is pressed
    if key==81 or key==113:
        break

#release the video capture object
webcam.release()

print("Code Completed")