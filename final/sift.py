import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import sklearn 

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split 

#Doc anh trong tep huan luyen
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('train'):
        for img_file in os.listdir(os.path.join('train',label)):
            img = cv2.imread(os.path.join('train',label,img_file))
            X.append(img)
            Y.append(label2id[label])
    return X,Y

#Gan nhan du lieu
label2id = {'peace':0, 'hello':1, 'thump':2, 'fist':3, 'right_hand':4}
X, Y = read_data(label2id)

#Trich xuat dac trung
def extr_sift_feature(X):
    image_descriptor =[]
    sift = cv2.xfeatures2d.SIFT_create()
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
x_train, x_test, y_train, y_test = train_test_split(X_features, Y, test_size=0.2)

svm = sklearn.svm.SVC(C = 10)
svm.fit(x_train,y_train)

print(svm.score(x_test, y_test))