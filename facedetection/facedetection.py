import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/LENOVO/Desktop/OpenCV-master/OpenCV-master/faces/'
#onlyfiles datatype is list and the datatype for onlyfiles[0] is string
#onlyfile contains the name of all the images
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
# Training_Data, Labels are empty lists
Training_Data, Labels = [], []
#value of i(integer) starts from 0 to 99 as per our 100 images,files (str) contains the name of the images
for i, files in enumerate(onlyfiles):
    #image_path contains the complete address for a image
    #eg-print(image_path)   C:/Users/LENOVO/Desktop/OpenCV-master/OpenCV-master/faces/user98.jpg
    image_path = data_path + onlyfiles[i]
    # datatype of images is numpy.ndarray as it contains the values of all pixels  eg [ 73  84 102 ... 141 141 141]
    # to see the array use print(images)
    # cv2.IMREAD_GRAYSCALE tells the compiler that we are reading a grayscale image by default is colo image
    # code for printing the last image that got into image.
    # while True:
    #     cv2.imshow("hey",images)
    #     if cv2.waitKey(1)==13:
    #         break
    # cv2.destroyAllWindows()
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   # training_data is a list datatype whereas images are np.ndarray types dtype represents that the values are unsigned integer of 8 bit
   # google says- uint8 is used unsigned 8 bit integer. And that is the range of pixel. We can't have pixel value more than 2^8 -1.
   # Therefore, for images uint8 type is used. Whereas double is used to handle very big numbers.
   # Training_Data.append((images)) also works fine as images are already of type np.ndarray test using print(Training_Data[0])
    Training_Data.append(np.asarray(images, dtype=np.uint8))
   #Labels is a list containing  the value of i [0,1,2...99]
    Labels.append(i)
   #converting labels to array
Labels = np.asarray(Labels, dtype=np.int32)
   #creating a model based on LBPHFaceRecognizer... but what it is?
model = cv2.face.LBPHFaceRecognizer_create()
   #training model using the images values and their labels.
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")


face_classifier = cv2.CascadeClassifier('C:/Users/LENOVO/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
        #     display_string = str(confidence)+'% Confidence it is user'
        # cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


        if confidence > 75:
            cv2.putText(image, "Welcome Ashish", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()


