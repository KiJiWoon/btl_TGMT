import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import sklearn 
import time
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def extr_sift_feature(X):
    image_descriptor =[]
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i],None)
        image_descriptor.append(des)
    return image_descriptor

def kmeans_bow(all_descriptor, num_cluster):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_cluster).fit(all_descriptor)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

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