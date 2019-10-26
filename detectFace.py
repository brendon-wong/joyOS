import numpy as np
import cv2
# path : /Users/uriyahannruiz/Documents/CalHacks6/emotionai/venv/cascades/data/haarcascade_frontalface_alt2.xml
# cv2.__file__ : /Users/uriyahannruiz/Documents/devProjects/env/faceRec_env/lib/python2.7/site-packages/cv2/cv2.so
# root: /Users/uriyahannruiz/Documents/devProjects/env/faceRec_env/lib/python2.7/site-packages/cv2/
faceCascade =  cv2.CascadeClassifier('/Users/uriyahannruiz/Documents/CalHacks6/emotionai/venv/cascades/data/haarcascade_frontalface_alt2.xml')
capture = cv2.VideoCapture(0)
while (True):
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #detecting gray
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        # region of interest(roi)
        print(x,y,w,h)

        # get the pixel values for that item as a square
        # (ycord_start - ycord_end, xcord_start - xcord_end)
        roiGray = gray[y:y+h, x:x+w] 
        roiColor = frame[y:y+h, x:x+w]
        imgItem = 'myImg.png'
        cv2.imwrite(imgItem, roiGray)

        # draw frame around object
        color = (255,0,0) #BGR 0-255
        stroke = 2
        endCordX = x + w
        endCordY = y + h
        cv2.rectangle(frame, (x,y), (endCordX, endCordY), color, stroke)

        # recognizer (not complete)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

