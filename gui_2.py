import Tkinter as tk
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
import wikipedia

from PIL import Image, ImageTk

clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")
cap = cv2.VideoCapture(0)
# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SURF_create()

root = tk.Tk()
lmain = tk.Label(root)
lmain2 = tk.Label(root)
pred = tk.Label(root)

def show1(root):
    kpts, des = sift.detectAndCompute(frame, None)
    test_features = np.zeros((1,k),"float32")
    words, distance = vq(des,voc)
    #print words,distance
    for w in words:
        test_features[0][w] += 1
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    idf = np.array(np.log((2.0) / (1.0*nbr_occurences + 1)), 'float32')
    # Scale the features
    test_features = stdSlr.transform(test_features)
    # Perform the predictions
    predictions =  [classes_names[i] for i in clf.predict(test_features)]
    pred["text"] = wikipedia.summary(predictions[0], sentences = 2)
    pt = (3, frame.shape[0] // 4)
    frame2 = cv2.flip(frame, 1)
    cv2.putText(frame2, predictions[0], pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
    img2 = Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)

def show_frame():
    _, frame = cap.read()
    frame1 = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    return frame

frame = show_frame()
button1 = tk.Button(master = root, text = "Recognize", command = lambda:show1(root))
button1.pack()
pred.pack()
lmain.pack({"side": "left"})
lmain2.pack({"side": "left"})
root.mainloop()
