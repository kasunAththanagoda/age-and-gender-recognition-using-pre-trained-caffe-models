#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
#os.chdir('D:\Python37\Projects\Gender-and-Age-Detection- Youtube\Gender-and-Age-Detection\models')


# In[33]:

#this function is for detecting the face and drawing a rectangle around the face
def detectFace(net,frame,confidence_threshold=0.7): # here confidence is the least expected acuracy of the identification
    frameOpencvDNN=frame.copy()
    print(frameOpencvDNN.shape)
    frameHeight=frameOpencvDNN.shape[0] #these sizes are from a array
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    #this is a function from cv2 dnn .it will input a image or a video and output a 4d array (blob) .which is ready to be processed
    #second parameter is a scale factor which is scaling it to quality that can be processed by cnn
    #third is the size which is expected by cnn
    #fourth is mean subtraction values.given mean values of red green and blue will b subtracted from the original image.this reduce the resolution and make easy to process
    #opencv assums colors in BGR .so it is converted
    #sixth is crop.
    net.setInput(blob) # now the video is processed in to numbers which suits cnn.blob will be inputed in to the neural network of the pre trained models
    detections=net.forward() #getting the  output
    #but we cant see that the face has been detected so we must draw the box
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2] # detection is a multi dimentional array .confidence variable is for the accuray
        if confidence>confidence_threshold:
            # these are the point values of the rectangle
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8) #drawing the rectangle around the face
            #first parameter is the image or the frame
            #then the points
            #then the color of the box in BNG format
            #then the line type
    return frameOpencvDNN,faceBoxes
        
    
faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

genderList=['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

faceNet=cv2.dnn.readNet(faceModel,faceProto) #readnet Read deep learning network represented in one of the supported formats
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(0) #getting the video input
padding=20
while cv2.waitKey(1)<0: #until u press the close button
    hasFrame,frame=video.read() # i thnik this is assigning the same method for both the variables
    if not hasFrame: #if there isno video or hasframe then brake
        cv2.waitKey()
        break
        
    resultImg,faceBoxes=detectFace(faceNet,frame) # returned two values are assigned to these two variables
    
    if not faceBoxes: #when the facebox is not detected
        print("No face detected")
    
    #for every second faceboxes will be detected    
    for faceBox in faceBoxes: #faceboxes is a  array with coordinate values
        #you can do this without the following face line
        #blob=cv2.dnn.blobFromImage(resultImg,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)] #this is used to get the proper face .the padding is used to reduce the unwanted area
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        #now the face is detected and coordinates are in the blob as a array
        genderNet.setInput(blob) #here the coordinates of face are inputed in to the gendernet which is a variable creatd from the pre trained models
        genderPreds=genderNet.forward()#then the output is taken fro the gendernet
        #print(genderPreds) #here the printed genderpreds are an array that contains properbilities for each category
        gender=genderList[genderPreds[0].argmax()] #the inde value of the array of maximum probabilty
        print(gender)
        
        ageNet.setInput(blob) #same as above
        agePreds=ageNet.forward()
        #print(agePreds)
        age=ageList[agePreds[0].argmax()]
        print(age)

        cv2.putText(resultImg,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)#adding the text to the image.thrid parameteris the location of the text.then the font.then font scal.then font colorthen thickness
        cv2.imshow("Detecting age and Gender",resultImg) #imshow() method is used to display an image in a window.first parmeter is a string to be o the top bar.and then the image to be shown
        
        
        if cv2.waitKey(33) & 0xFF == ord('q'): #if the button is hold for 33 seconds then break the loop
            break
            
cv2.destroyAllWindows()
        


# In[ ]:




