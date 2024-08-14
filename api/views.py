from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from json import JSONEncoder
import json
import cv2
import mediapipe as mp
import urllib.request
import numpy as np

class handapi(APIView):
    allowed_methods = ['POST']

    def __init__(self):
        self.mp_hands = mp.solutions.hands

    def post(self, request, *args, **kwargs):
        # Create a face mesh object

          responseData = {"multiHandedness":None,"multiHandLandmarks":None,"scaleFactor":1,"offsetX":0,"offsetY":0}
          scaleFactor = 1
          offsetX = 0
          offsetY = 0
          requestData = request.data
          badLandmarks = requestData["multiHandLandmarks"]

          try:
               imageData = request.data["image"]
               if imageData != None:
                    with self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.5) as hands:
                    # Read image file with cv2 and convert from BGR to RGB
                         req = urllib.request.Request(imageData)
                         buffer = urllib.request.urlopen(req)
                         bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
                         image = cv2.imdecode(bytes_as_np_array,cv2.IMREAD_UNCHANGED)
                         imageCopy = image.copy()
                         #image = cv2.flip(image, 1) 

                         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                         hand_found = bool(results.multi_hand_landmarks)
                         landmarks = []
                         if hand_found:
                                goodXCoords = []
                                goodYCoords = []
                         
                                for i in results.multi_hand_landmarks[0].landmark:
                                    landmarks.append({"x":i.x,"y":i.y,"z":i.z})
                                    goodX = int(i.x*image.shape[1])
                                    goodY = int(i.y*image.shape[0])
                                    cv2.circle(imageCopy, (goodX,goodY), 2, (0,255,0), -1)    

                                    goodXCoords.append(goodX)
                                    goodYCoords.append(goodY)
                                
                                goodBboxTopLeftX = int(min(goodXCoords))
                                goodBboxTopLeftY = int(min(goodYCoords))
                                goodBboxBottomRightX = int(max(goodXCoords))
                                goodBboxBottomRightY = int(max(goodYCoords))
                                goodBboxWidth = goodBboxBottomRightX - goodBboxTopLeftX
                                goodBboxHeight = goodBboxBottomRightY - goodBboxTopLeftY
                                goodBboxCenterX = goodBboxTopLeftX + int(goodBboxWidth/2)
                                goodBboxCenterY = goodBboxTopLeftY + int(goodBboxHeight/2)

                                cv2.rectangle(imageCopy, (goodBboxTopLeftX, goodBboxTopLeftY),
                                            (goodBboxBottomRightX, goodBboxBottomRightY),(0,255,0) , 2) 

                                badXCoords = []
                                badYCoords = []
                                
                                if badLandmarks != None:

                                    for lmark in badLandmarks:

                                        badX = int(lmark['x']*image.shape[1])
                                        badY = int(lmark['y']*image.shape[0])
                                        cv2.circle(imageCopy, (badX,badY), 2, (0,0,255), -1)    

                                        badXCoords.append(badX)
                                        badYCoords.append(badY)

                                    badBboxTopLeftX = int(min(badXCoords))
                                    badBboxTopLeftY = int(min(badYCoords))
                                    badBboxBottomRightX = int(max(badXCoords))
                                    badBboxBottomRightY = int(max(badYCoords))
                                    badBboxWidth = badBboxBottomRightX - badBboxTopLeftX
                                    badBboxHeight = badBboxBottomRightY - badBboxTopLeftY
                                    badBboxCenterX = badBboxTopLeftX + badBboxWidth/2
                                    badBboxCenterY = badBboxTopLeftY + badBboxHeight/2

                                    offsetX = goodBboxCenterX - badBboxCenterX
                                    offsetY = goodBboxCenterY - badBboxCenterY
                                    scaleFactor = (goodBboxWidth * goodBboxHeight)/(badBboxWidth*badBboxHeight)
            
                                    cv2.rectangle(imageCopy, (badBboxTopLeftX, badBboxTopLeftY),
                                                (badBboxBottomRightX, badBboxBottomRightY),(0,0,255) ,2) 
                                    
                                cv2.imwrite("handboxes.jpg", imageCopy) 
                                
                                multiHandedness = results.multi_handedness[0].classification
                                multiHandedness = [{"index":multiHandedness[0].index,"score":multiHandedness[0].score,"label":multiHandedness[0].label}]
                                responseData = {"multiHandedness":multiHandedness,"multiHandLandmarks":[landmarks],"scaleFactor":scaleFactor,
                                                "offsetX":offsetX,"offsetY":offsetY}
                                
                                #JSONEncoder().encode({"multiHandLandmarks":landmarks})  
                                                
          except Exception as err:
               print("Hand API error occured ",err)
                
          return Response(responseData,content_type="application/json", status=status.HTTP_200_OK)
        
    def get(self, request, *args, **kwargs):
        return Response("",content_type="application/json", status=status.HTTP_200_OK)
    
class faceapi(APIView):
    allowed_methods = ['POST']

    def __init__(self):
         self.mp_face_mesh = mp.solutions.face_mesh

    def post(self, request, *args, **kwargs):
          
          responseData = {"multiFaceLandmarks":None,"scaleFactor":1,"offsetX":0,"offsetY":0}
          scaleFactor = 1
          offsetX = 0
          offsetY = 0      

          try:
            requestData = request.data
            imageData = requestData["image"]
            badLandmarks = requestData["multiFaceLandmarks"]
            if imageData != None:
               # Create a face mesh object
                with self.mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:
                    
                    req = urllib.request.Request(imageData)
                    buffer = urllib.request.urlopen(req)
                    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
                    image = cv2.imdecode(bytes_as_np_array,cv2.IMREAD_UNCHANGED)
                    imageCopy = image.copy()
                    image = cv2.flip(image, 1) 

                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    face_found = bool(results.multi_face_landmarks)
                    landmarks = []

                    if face_found:
                        goodXCoords = []
                        goodYCoords = []
                    
                        for i in results.multi_face_landmarks[0].landmark:
                            landmarks.append({"x":i.x,"y":i.y,"z":i.z})
                            goodXCoords.append(i.x*image.shape[1])
                            goodYCoords.append(i.y*image.shape[0])
                        
                        goodBboxTopLeftX = int(min(goodXCoords))
                        goodBboxTopLeftY = int(min(goodYCoords))
                        goodBboxBottomRightX = int(max(goodXCoords))
                        goodBboxBottomRightY = int(max(goodYCoords))
                        goodBboxWidth = goodBboxBottomRightX - goodBboxTopLeftX
                        goodBboxHeight = goodBboxBottomRightY - goodBboxTopLeftY
                        goodBboxCenterX = goodBboxTopLeftX + int(goodBboxWidth/2)
                        goodBboxCenterY = goodBboxTopLeftY + int(goodBboxHeight/2)

                        cv2.rectangle(imageCopy, (goodBboxTopLeftX, goodBboxTopLeftY),
                                       (goodBboxBottomRightX, goodBboxBottomRightY),(0,255,0) , 2) 

                        badXCoords = []
                        badYCoords = []
                        
                        if badLandmarks != None:
                            for landmark in badLandmarks:
                                badXCoords.append(landmark['x']*image.shape[1])
                                badYCoords.append(landmark['y']*image.shape[0])

                            badBboxTopLeftX = int(min(badXCoords))
                            badBboxTopLeftY = int(min(badYCoords))
                            badBboxBottomRightX = int(max(badXCoords))
                            badBboxBottomRightY = int(max(badYCoords))
                            badBboxWidth = badBboxBottomRightX - badBboxTopLeftX
                            badBboxHeight = badBboxBottomRightY - badBboxTopLeftY
                            badBboxCenterX = badBboxTopLeftX + badBboxWidth/2
                            badBboxCenterY = badBboxTopLeftY + badBboxHeight/2

                            offsetX = goodBboxCenterX - badBboxCenterX
                            offsetY = goodBboxCenterY - badBboxCenterY
                            scaleFactor = goodBboxWidth/badBboxWidth

                            cv2.rectangle(imageCopy, (badBboxTopLeftX, badBboxTopLeftY),
                                        (badBboxBottomRightX, badBboxBottomRightY),(255,0,0) ,2) 
                            
                        cv2.imwrite("faceboxes.jpg", imageCopy) 

                        
                        responseData = {"multiFaceLandmarks":[landmarks],"scaleFactor":scaleFactor,"offsetX":offsetX,"offsetY":offsetY} 

                    #JSONEncoder().encode({"multiHandLandmarks":landmarks})       
          except Exception as e:
            responseData = {}
            print(e)
          return Response(responseData,content_type="application/json", status=status.HTTP_200_OK)
                                            
    def get(self, request, *args, **kwargs):
        return Response("",content_type="application/json", status=status.HTTP_200_OK)