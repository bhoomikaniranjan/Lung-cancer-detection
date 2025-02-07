import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten
from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk

class LCD_CNN:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        img4 = Image.open(r"Images\Lung-Cancer-Detection.jpg")
        img4 = img4.resize((1006, 500))
        self.photoimg4 = ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=50, width=1006, height=500)

        title_lbl = Label(text="Lung Cancer Detection", font=("Bradley Hand ITC", 30, "bold"), bg="black", fg="white")
        title_lbl.place(x=0, y=0, width=1006, height=50)

        self.b1 = Button(text="Import Data", cursor="hand2", command=self.import_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b1.place(x=80, y=130, width=180, height=30)
        self.b2 = Button(text="Pre-Process Data", cursor="hand2", command=self.preprocess_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b2.place(x=80, y=180, width=180, height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")
        self.b3 = Button(text="Train Data", cursor="hand2", command=self.train_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b3.place(x=80, y=230, width=180, height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

    def import_data(self):
        self.dataDirectory = 'sample_images/'
        self.lungPatients = os.listdir(self.dataDirectory)
        self.labels = pd.read_csv('stage1_labels.csv', index_col=0)
        self.size = 10
        self.NoSlices = 5
        messagebox.showinfo("Import Data", "Data Imported Successfully!")
        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")

    def preprocess_data(self):
        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if count < self.NoSlices:
                    yield l[i:i + n]
                    count += 1

        def mean(l):
            return sum(l) / len(l)

        def dataProcessing(patient, labels_df, size=10, noslices=5, visualize=False):
            label = labels_df._get_value(patient, 'cancer')
            path = self.dataDirectory + patient
            slices = [dicom.dcmread(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

            chunk_sizes = math.floor(len(slices) / noslices)
            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list(map(mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            label = np.array([0, 1]) if label == 1 else np.array([1, 0])
            return np.array(new_slices), label

        imageData = []
        for num, patient in enumerate(self.lungPatients):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                img_data, label = dataProcessing(patient, self.labels, size=self.size, noslices=self.NoSlices)
                print(f"Patient: {patient}, img_data shape: {img_data.shape}, label shape: {label.shape}")
                imageData.append([img_data, label, patient])
            except KeyError as e:
                print('Data is unlabeled')

        np.save('imageDataNew-{}-{}-{}.npy'.format(self.size, self.size, self.NoSlices), np.array(imageData, dtype=object))
        messagebox.showinfo("Pre-Process Data", "Data Pre-Processing Done Successfully!")
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        imageData = np.load('imageDataNew-10-10-5.npy', allow_pickle=True)
        trainingData = imageData[0:45]
        validationData = imageData[45:50]

        training_data = Label(text="Total Training Data: " + str(len(trainingData)), font=("Times New Roman", 13, "bold"), bg="black", fg="white")
        training_data.place(x=750, y=150, width=200, height=18)
        validation_data = Label(text="Total Validation Data: " + str(len(validationData)), font=("Times New Roman", 13, "bold"), bg="black", fg="white")
        validation_data.place(x=750, y=190, width=200, height=18)

        size = 10
        NoSlices = 5

        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(size, size, NoSlices, 1), padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        X_train = np.array([data[0] for data in trainingData]).reshape(-1, size, size, NoSlices, 1)
        Y_train = np.array([data[1] for data in trainingData])
        X_val = np.array([data[0] for data in validationData]).reshape(-1, size, size, NoSlices, 1)
        Y_val = np.array([data[1] for data in validationData])

        model.fit(X_train, Y_train, epochs=35, validation_data=(X_val, Y_val))

# For GUI
if __name__ == "__main__":
    root = Tk()
    obj = LCD_CNN(root)
    root.mainloop()
