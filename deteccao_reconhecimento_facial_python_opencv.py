#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

#get paths of each file in folder named Imagens
#Images here contains my data (folders of various persons)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images('\\images'))

# initialize the list of knwon encodings and knwon names
knownEncodings = []
knownNames = []

#loop over the images paths
for (i, imagePath) in enumerate(imagePaths):
    
    #extract the person name from the image path
    # "images","Nicolas Cage","NC1.jpg" - retira (Nicolas Cage)
    # "images", "Robert Downey Jr","RDJ1.jpg"- retira (Robert Downey Jr)
    print('[INFO] processing image {}/{}'.format(i + 1, len(imagePaths)))
    #name = imagePath.split(os.path.sep)[-2].split(",")[0]
    name = imagePath.split(os.path.sep)[-2]
    print(imagePaths[i])
        
    #load the input image and convert it from BGR (OpenCV ordering) to dlib ordenig (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Use face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb, model='cnn')
    
    #compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes, model='large')
    
    #loop over the encodings
    for encoding in encodings:
        
        # add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
        
    #save encodings along with their names in dictionary data
    print("[INFO] serializing encodings...")
    data = {'encodings': knownEncodings, 'names': knownNames}
    
    #use pickle to save into a file for later use
    f = open('face_enc', 'wb')
    f.write(pickle.dumps(data))
    f.close()
    


# In[ ]:


# Importing the necessary packages
#import face_recognition
#import imutils
import pickle
import time
#import cv2
#import os

# find path of xml containing haarcascade file
#cascPathface = os.path.dirname(
#    cv2.__file__) + "/data/haarcascades/haarcascade_frontalface_default.xml"

# load the haarcascade in the cascade classifier
#Alterei pois o código não estava encontrando o caminho aonde estava armazenado o haarcascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#faceCascade = cv2.CascadeClassifier(cascPathface)


# load the known face and embeddings saved in last file
data = pickle.loads(open('face_enc', 'rb').read())

#print('Streaming Started')
video_capture = cv2.VideoCapture(0)
    
# loop over frames from from the video file stream
while (True):

    # grab the frame from the threaded video stream
    ret, frame = video_capture.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor = 1.04, 
                                         minNeighbors = 3,
                                         minSize = (60,60),
                                         flags = cv2.CASCADE_SCALE_IMAGE)
    
    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # the facial embedding for face in input
    encoding = face_recognition.face_encodings(rgb)
    names = []
    
    # loop over the facial embeddings incase we have multiple embeddings for multiple faces
    for encoding in encodings:
        
        # compare encodings with encodings in data ['encodings']
        # Matches contain array with boolean values and True for the embeddingd it matches closely and False for rest
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        
        # set name = inknown if no encoding matches
        name = 'Unknwon'
        
        # check to see we have found a match
        if True in matches:
            
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches)if b]
            counts = {}
            
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                
                # Check the names at respective indexes we stored in matchedIdxs
                name = data['names'][i]
                
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                
                # set name which has highest count
                name = max(counts, key=counts.get)
                
            # update the list of names
            names.append(name)
            
            # loop over the recognized faces
            for ((x, y, w, h),name) in zip(faces, names):
                
                # rescale the face coordinates
                # draw the predicated face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
video_capture.release()
cv2.destroyAllWindows()
    


# In[ ]:




