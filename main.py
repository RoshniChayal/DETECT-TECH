from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


class Detect:
    def __init__(self):
        self.gen = None
        self.der = None
        self.emotion_model = Sequential()
        self.emotion_model.add(Conv2D(32, kernel_size=(
            3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.emotion_model.add(
            Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))
        self.emotion_model.add(
            Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(
            Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))
        self.emotion_model.add(Flatten())
        self.emotion_model.add(Dense(1024, activation='relu'))
        self.emotion_model.add(Dropout(0.5))
        self.emotion_model.add(Dense(7, activation='softmax'))
        self.emotion_model.load_weights('model.h5')
        cv2.ocl.setUseOpenCL(False)
        self.emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ",
                             3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
        self.emoji_dist = {0: "emojis/angry.png", 2: "emojis/disgusted.png", 2: "emojis/fearful.png",
                           3: "emojis/happy.png", 4: "emojis/neutral.png", 5: "/emojis/sad.png", 6: "emojis/surprised.png"}
        self.show_text = [0]
        self.root = Tk()
        self.gui()
        self.root.mainloop()

    def gui(self):
        self.inputVideo()
        self.outputVideo()
        self.outputGender()
        self.root.title("Detect Tech")
        heading = Label(self.root, text="Detect Tech",
                        font=("Arial bold", 100), padx=100)
        label1 = Label(self.root, text="Input", font=("Arial", 50))
        label2 = Label(self.root, text="Output", font=("Arial", 50))
        button1 = Button(self.root, text="Go Again", font=(
            "Arial bold", 30), fg='green', borderwidth=5, padx=20, command=self.gui)
        button2 = Button(self.root, text="Quit", font=(
            "Arial bold", 30), fg='red', borderwidth=5, padx=20, command=self.root.destroy)
        canvas1 = Label(self.root, height=300, width=300)
        canvas2 = Label(self.root, height=300, width=300)
        emotion_output = Label(
            self.root, text=self.emotion_dict[self.show_text[0]], font=("Arial", 15))
        gender_output = Label(
            self.root, text=self.gen, font=("Arial", 15))
        age_output = Label(
            self.root, text=self.der, font=("Arial", 15))
        iimg = cv2.imread("Input.png")
        oimg = cv2.imread("Output.png")
        width = 290
        height = 290
        dim = (width, height)
        inp_resized = cv2.resize(iimg, dim, interpolation=cv2.INTER_AREA)
        inp_img = Image.fromarray(inp_resized)
        out_resized = cv2.resize(oimg, dim, interpolation=cv2.INTER_AREA)
        out_img = Image.fromarray(out_resized)
        self.inputtk = ImageTk.PhotoImage(image=inp_img)
        canvas1.configure(image=self.inputtk)
        self.outputtk = ImageTk.PhotoImage(image=out_img)
        canvas2.configure(image=self.outputtk)
        heading.grid(row=0, column=0, columnspan=3)
        label1.grid(row=1, column=0)
        label2.grid(row=1, column=2)
        canvas1.grid(row=2, column=0)
        canvas2.grid(row=2, column=2)
        emotion_output.grid(row=3, column=2)
        gender_output.grid(row=4, column=2)
        age_output.grid(row=5, column=2)
        button1.grid(row=6, column=0)
        button2.grid(row=6, column=2)

    def inputVideo(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.root.destroy()
            print("Can't open the camera")
        i = 0
        while(i < 20):
            flag, frame = capture.read()
            i += 1
        cv2.imwrite("Input.png", frame)
        bounding_box = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            self.show_text[0] = maxindex

    def outputVideo(self):
        frame1 = cv2.imread(self.emoji_dist[self.show_text[0]])
        pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        cv2.imwrite("Output.png", pic)

    def outputGender(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image')

        args = parser.parse_args()

        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-3)', '(4-6)', '(7-13)', '(14-22)',
                   '(23-34)', '(35-45)', '(46-56)', '(57-100)']
        genderList = ['Male', 'Female']

        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)

        video = cv2.VideoCapture(args.image if args.image else 0)
        padding = 20
        i = 0
        while i < 20:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey()
                break

            resultImg, faceBoxes = self.highlightFace(faceNet, frame)
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]-padding):
                             min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                self.gen = gender

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                self.der = age

                cv2.putText(resultImg, f'{gender}, {age}', (
                    faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            i += 1

    def highlightFace(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                     104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3]*frameWidth)
                y1 = int(detections[0, 0, i, 4]*frameHeight)
                x2 = int(detections[0, 0, i, 5]*frameWidth)
                y2 = int(detections[0, 0, i, 6]*frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                              (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, faceBoxes


if __name__ == "__main__":
    obj = Detect()
