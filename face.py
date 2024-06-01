'''#let us understand how FaceDetection
import cv2
#pretrained classifier-->haarcascade classifier(frontal face)
img=cv2.imread("ena.jpg")
#then we  will access pretrained human faces cascade classifier
cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#print(cascade)
#we will use above cascade classifier on our image to detect the face
g=cascade.detectMultiScale(img,1.3,3)
#both scaling factors and neighbours values to be adjusted
#print(g)#if you get empty tuple there is no frontal face detected
for (x,y,w,h) in g:
    #we will use rectangle as bounding box
    d=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
    #print(d)
    #now we will add text over image
    cv2.putText(d,"Ee nagaraniki emaindhi",(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),5)
    #finally we will resize the image and display it
    resized=cv2.resize(d,(600,600))
    #display the image
cv2.imshow('facedetect',img)
cv2.waitKey()
cv2.destroyAllWindows()'''

import cv2
#access cascade classifier
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#now we will use camera as our source
#internal camera - 0,external webcam - 1
cap = cv2.VideoCapture(0)
#print(cap) #it should return an object to check camera is working/not
#we will set the width and height for the video frame
cap.set(3,1500) #set the width
cap.set(4,1600) #set the height
#we will perform iteration from the video capture and detect face from the camera
while True:
    ret,img = cap.read() #to read the image
    #then we will flip the image to get the proper face region deteced from the camera
    img = cv2.flip(img,1)
    #then we need to go for colorconversion cvtColor
    gray = cv2.cvtColor(img,7) #mention the colorcode
    #detect the faces
    faces = cascade.detectMultiScale(gray,1.5,5)
    #print(faces)
    for (x,y,w,h) in faces:
        #we will draw the rectangle box for face detected
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #picking the region of interest
        roi_gray = gray[y:y+h,x:x+w]
        roi_col = img[y:y+h,x:x+w]
    #if you want you can add text using putText
    cv2.imshow("Me",img) #getting the possible match finally
    #here before releasing the camera windows we will map ASCII values
    k = cv2.waitKey(30) & 0xff #this is for matching encoding value
    if k==27: #here 27 is the ASCII value for Escape button
        break
cap.release() #finally once user presses Esc button capture to be released
cv2.destroyAllWindows()                                   
                                

