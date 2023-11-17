import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
import numpy as np
import cv2

from sift_des import extr_sift_feature, create_features_bow, kmeans_bow

# Load model
num_cluster = 100
BOW = joblib.load(open('model/model.pkl','rb'))
SVM = joblib.load('svm_model.sav')
label2id = {'Peace':0, 'Hello':1, 'One':2, 'Fist':3, 'Ok':4}
# Hàm dự đoán ảnh
def predict_image(image_path):
    # Đọc ảnh và tiền xử lý
    image = cv2.imread(image_path)
    img_test = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hei = cv2.equalizeHist(img_test)
    img = [hei]
    img_sift_feature = extr_sift_feature(img)
    img_bow_feature = create_features_bow(img_sift_feature,BOW,num_cluster)
    predicted = SVM.predict(img_bow_feature)
    for key, value in label2id.items():
        if value == predicted[0]:
            predicted_class = key
    
    return predicted_class

# Hàm chọn ảnh và hiển thị kết quả
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class = predict_image(file_path)

        # Hiển thị kết quả
        result_label.config(text=f'Predicted Class: {predicted_class}')

        # Hiển thị ảnh
        image = Image.open(file_path).resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

# Tạo giao diện đồ họa
root = tk.Tk()
root.title("Image Prediction")

# Button để chọn ảnh
choose_button = tk.Button(root, text="Choose Image", command=choose_image)
choose_button.pack(pady=10)

# Label để hiển thị kết quả
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Label để hiển thị ảnh
image_label = tk.Label(root)
image_label.pack(pady=10)

# Chạy ứng dụng Tkinter
root.mainloop()