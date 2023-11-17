import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import sklearn 
import time
import seaborn as sns

from cvzone.HandTrackingModule import HandDetector
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

#Doc anh trong tep huan luyen
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('train'):
        for img_file in os.listdir(os.path.join('train',label)):
            img = cv2.imread(os.path.join('train',label,img_file))
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img,(255,255))
            gray_img = cv2.equalizeHist(gray_img)
            X.append(gray_img)
            Y.append(label2id[label])
    return X,Y

#Gan nhan du lieu
label2id = {'Peace':0, 'Hello':1, 'One':2, 'Fist':3, 'Ok':4}
X, Y = read_data(label2id)
#Trich xuat dac trung
def extr_sift_feature(X):
    image_descriptor =[]
    sift = cv2.SIFT_create()
    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i],None)
        image_descriptor.append(des)
    return image_descriptor

images_descriptor = extr_sift_feature(X)

#print(len(images_descriptor))

all_descriptor =[]
for descriptor in images_descriptor:
    if descriptor is not None:
        for des in descriptor:
            all_descriptor.append(des)

#print(all_descriptor)

def kmeans_bow(all_descriptor, num_cluster):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_cluster).fit(all_descriptor)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_cluster = 100

if not os.path.isfile('model/model.pkl'):
    BOW = kmeans_bow(all_descriptor, num_cluster)
    pickle.dump(BOW, open('model/model.pkl','wb'))
else:
    BOW = pickle.load(open('model/model.pkl','rb'))


def create_features_bow(images_descriptor, BOW, num_cluster):
    X_features = []
    for i in range(len(images_descriptor)):
        features = np.array([0] * num_cluster)

        if images_descriptor[i] is not None:
            distance = cdist(images_descriptor[i],BOW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] +=1
        X_features.append(features)
    return X_features

X_features = create_features_bow(images_descriptor,BOW,num_cluster)

#xay dung model
x_train = []
y_train = []
x_test = []
y_test = []
x_train, x_test, y_train, y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

SVM = SVC(kernel='rbf',C = 12)
SVM.fit(x_train,y_train)


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10,20,25,30,35, 100, 1000]}

grid_model = GridSearchCV(SVM,param_grid,cv=10)
grid_model.fit(x_train,y_train)


# # In giá trị C tốt nhất
print("Best C:", grid_model.best_params_)
print("Best Scorce:", grid_model.best_score_)






filename = 'svm_model.sav'
pickle.dump(SVM, open(filename, 'wb'))

# print(SVM.score(x_test, y_test))
# y_pred = SVM.predict(x_test)
# cm = confusion_matrix(y_test,y_pred)
# print(cm)
# print(classification_report(y_test,y_pred))

# param_range = np.logspace(-3, 3, 7)
# train_scores, test_scores = validation_curve(sklearn.svm.SVC(), x_train, y_train, param_name='C', param_range=param_range, cv=5)

# plt.figure(figsize=(10, 6))
# plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Training score')
# plt.semilogx(param_range, np.mean(test_scores, axis=1), label='Validation score')
# plt.xlabel('C (Regularization parameter)')
# plt.ylabel('Score')
# plt.legend()
# plt.show()


# x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
# y = [20, 25, 30, 40, 45, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]



# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.9,maxHands=1)

# while True:
#     ret, frame = cap.read()
#     hands, frame = detector.findHands(frame, draw=False)

#     numeric = 0

#     if hands:

#         x,y,w,h = hands[0]['bbox']
#         fr = frame[y-20: (y+h)+35,x-20: (x+w)+35]
#         if fr.size != 0:
#             #frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#             cv2.imwrite('./valid/hand_gesture_{}.jpg'.format(numeric),frame[y-20: (y+h)+35, x-20: (x+w)+35])
#         # cv.imwrite('./hand_cut/hand_gesture.jpg',frame[y-20: (y+h)+35, x-20: (x+w)+35])
            

#         cv2.rectangle(frame, (x-20,y-20), ( (x+w) + 35 , (y+h) + 35 ), (0,255,0), 3)

#         numeric = numeric + 1

#     if ret == True:
        
#         file = 'valid'
#         for i in os.listdir(file):
#             detect = os.path.join(file, i)
#             img_test = cv2.imread(detect)
#             img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
#             hei = cv2.equalizeHist(img_test)
#             img = [hei]
#             img_sift_feature = extr_sift_feature(img)
#             img_bow_feature = create_features_bow(img_sift_feature,BOW,num_cluster)
#             img_predict = SVM.predict(img_bow_feature)
#             print(img_predict)
        

#             for key, value in label2id.items():
#                 if value == img_predict[0]:
#                     cv2.putText(frame,key, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
#         cv2.imshow("result", frame)

        

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break 


# vid.release()
# cv2.destroyAllWidows()