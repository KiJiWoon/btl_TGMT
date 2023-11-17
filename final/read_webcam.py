import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import sklearn 
import time
import seaborn as sns
import joblib

from cvzone.HandTrackingModule import HandDetector
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sift_des import extr_sift_feature, create_features_bow, kmeans_bow

#Doc anh trong tep huan luye

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 40, 45, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]

num_cluster = 100
BOW = joblib.load(open('model/model.pkl','rb'))
SVM = joblib.load('svm_model.sav')
label2id = {'Peace':0, 'Hello':1, 'One':2, 'Fist':3, 'Ok':4}

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.9,maxHands=1)

while True:
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame, draw=False)

    numeric = 0

    if hands:

        x,y,w,h = hands[0]['bbox']
        fr = frame[y-20: (y+h)+35,x-20: (x+w)+35]
        if fr.size != 0:
            #frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./valid/hand_gesture_{}.jpg'.format(numeric),frame[y-20: (y+h)+35, x-20: (x+w)+35])
        # cv.imwrite('./hand_cut/hand_gesture.jpg',frame[y-20: (y+h)+35, x-20: (x+w)+35])
            

        cv2.rectangle(frame, (x-20,y-20), ( (x+w) + 35 , (y+h) + 35 ), (0,255,0), 3)

        numeric = numeric + 1

    if ret == True:
        
        file = 'valid'
        for i in os.listdir(file):
            detect = os.path.join(file, i)
            img_test = cv2.imread(detect)
            img_test = cv2.flip(img_test,1)
            img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
            hei = cv2.equalizeHist(img_test)
            img = [hei]
            img_sift_feature = extr_sift_feature(img)
            img_bow_feature = create_features_bow(img_sift_feature,BOW,num_cluster)
            img_predict = SVM.predict(img_bow_feature)
            print(img_predict)
        

            for key, value in label2id.items():
                if value == img_predict[0]:
                    cv2.putText(frame,key, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imshow("result", frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 


vid.release()
cv2.destroyAllWidows()