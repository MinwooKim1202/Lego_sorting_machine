import cv2
import threading
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit, QCheckBox, QPushButton, QHBoxLayout, QVBoxLayout)
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer

import matplotlib
import imutils
import timeit
import numpy as np
import serial
import time
import json
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Model, load_model

port = '/dev/ttyUSB0' # 시리얼 포트
baud = 9600 # 시리얼 보드레이트(통신속도)

msg = {'conveyor_step' : 0, 'sorting_step' : 0}
end_str = '\n'
memory = deque()
img_list = [0, 0]

model = tf.saved_model.load('LegoNet_V7_FP16')

lock = threading.Lock()

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.opencv_apply_flag = False
        self.Dnn_infer_apply_flag = False
        self.run_flag = False
        self.morph_flag = False


    def initUI(self):
        self.Name_label = QLabel('Machine Vision System V1')
        self.live_checkbox = QCheckBox('Live On/Off')
        self.serial_btn = QPushButton('Serial Connect')
        self.model_load_btn = QPushButton('Model Load')
        self.Opencv_checkbox = QCheckBox('Opencv On/OFF')
        self.Dnn_infer_checkbox = QCheckBox('DNN Infer On/OFF')
        self.image_label = QLabel('image view')
        self.edge_label = QLabel('Edge view')
        self.detect_label0 = QLabel('detect view0')
        self.detect_label1 = QLabel('detect view1')
        self.detect_label2 = QLabel('detect view2')
        self.detect_label3 = QLabel('detect view3')
        self.detect_label4 = QLabel('detect view4')
        self.detect_label5 = QLabel('detect view5')

        self.logwindow = QTextEdit()
        self.logwindow.setAcceptRichText(False)
        self.logwindow.setPlainText("Log start !!!")


        Name_label_font = self.Name_label.font()
        Name_label_font.setBold(True)
        Name_label_font.setPointSize(20)
        self.Name_label.setFont(Name_label_font)

        self.image_label.resize(640, 480)
        self.edge_label.resize(640, 480)
        self.detect_label0.resize(100, 100)
        self.detect_label1.resize(100, 100)
        self.detect_label2.resize(100, 100)
        self.detect_label3.resize(100, 100)
        self.detect_label4.resize(100, 100)
        self.detect_label5.resize(100, 100)
        
        base_img = cv2.imread('keras.jpg')
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB) 
        base_img = cv2.resize(base_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        base_h, base_w, base_c = base_img.shape
        base_qImg = QtGui.QImage(base_img.data, base_w, base_h, base_w*base_c, QtGui.QImage.Format_RGB888)
        
        base_pixmap = QtGui.QPixmap.fromImage(base_qImg)
        self.image_label.setPixmap(base_pixmap)
        
        grid = QGridLayout()
        grid.addWidget(self.detect_label0, 0,0)
        grid.addWidget(self.detect_label1, 0,1)
        grid.addWidget(self.detect_label2, 0,2)
        grid.addWidget(self.detect_label3, 1,0)
        grid.addWidget(self.detect_label4, 1,1)
        grid.addWidget(self.detect_label5, 1,2)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.Name_label)
        hbox1.addStretch(2)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.live_checkbox)
        hbox2.addWidget(self.serial_btn)
        hbox2.addWidget(self.model_load_btn)
        hbox2.addWidget(self.Opencv_checkbox)
        hbox2.addWidget(self.Dnn_infer_checkbox)
        hbox2.addStretch(2)
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.image_label)
        hbox3.addWidget(self.logwindow)

        
        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.edge_label)
        hbox4.addLayout(grid)

        vbox = QVBoxLayout()
        vbox.addStretch(0.1)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addStretch(1)
    
	
        self.setLayout(vbox)
    
        self.live_checkbox.stateChanged.connect(self.change_mode)	
        self.serial_btn.clicked.connect(self.serial_connect)
        self.model_load_btn.clicked.connect(self.model_load)
        self.Opencv_checkbox.stateChanged.connect(self.opencv_apply)
        self.Dnn_infer_checkbox.stateChanged.connect(self.Dnn_infer_apply)

        self.setWindowTitle('Machine Vision System')
        self.setGeometry(300, 100, 1000, 900)
        self.show()


    def gstreamer_pipeline(self,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=20,
    flip_method=0,
):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
            capture_width, 
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
            )
        )

    def clamp(self, val):
        if val < 0:
            val = 0
        return val

    def saturare(frame):
        saturate_val = 70
        imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(imghsv)
        s = s + saturate_val
        s = numpy.clip(s, 0, 255)
        imghsv = cv2.merge([h, s, v])
        imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        return imgrgb

    def video_run(self):

        global running

        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))



        while running:
            self.run_flag = True
            lock.acquire()
            
            ret, frame = cap.read()

            start_t = timeit.default_timer()
            if not ret:
                print("영상 종료")
                break

            video_fps = cv2.CAP_PROP_FPS

            frame = imutils.resize(frame, width=480)
    #frame = imutils.rotate(frame, angle=-90)
    #frame = frame[0:400, 160:560]
            origin_frame = frame.copy()

            if self.opencv_apply_flag == True:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                #L = lab[:, :, 0]
                #med_L = cv2.medianBlur(L, 99)
                #invert_L = cv2.bitwise_not(med_L)
                #composed = cv2.addWeighted(gray, 0.6, invert_L, 0.4, 0)
                #blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel1)  
                
                binary = cv2.adaptiveThreshold(morph_grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,45,4)
                median_blur = cv2.medianBlur(binary, 7)

                
                morph_close = cv2.morphologyEx(median_blur, cv2.MORPH_CLOSE, kernel1) 

                contours, _ = cv2.findContours(morph_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if (len(contours) > 0):
                    margin = 10
                    real_contours = contours[0]
                    max = 0
                    for i in range(0, len(contours)):
                        area = cv2.contourArea(contours[i])
                        if max < area:
                            max = area
                            real_contours = contours[i]
                    cnt = real_contours
                    x, y, w, h = cv2.boundingRect(cnt)

                    if w > h:  # 물체가 rect 중앙에 오게함
                        y = int(y - ((w - h) / 2))
                        h = int(h + (((w - h) / 2) * 2))
                    else:
                        x = int(x - ((h - w) / 2))
                        w = int(w + (((h - w) / 2) * 2))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    

                    if w > 30 and h > 30:
                        crop_fist_x = self.clamp(x - margin)
                        crop_fist_y = self.clamp(y - margin)
                        crop_twice_x = self.clamp(x + w + margin)
                        crop_twice_y = self.clamp(y + h + margin)
                        cropped_img = origin_frame[crop_fist_y: crop_twice_y, crop_fist_x: crop_twice_x]
                        cropped_img = cv2.resize(cropped_img, dsize=(88, 88), interpolation=cv2.INTER_LINEAR)
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        img_clahe = clahe.apply(cropped_img)

                        memory.append(cropped_img)
                        
                        if len(memory) > 7:

                        if abs(cropped_img.shape[0] - cropped_img.shape[1]) > 10: # width 와 height 가 10이상 차이나면 저장 x
                            continue

                        if self.Dnn_infer_apply_flag == True:
                            cropped_img = cropped_img / 255
                            merge_img = cropped_img.reshape((1, 88, 88, 1))

                            infer = model.signatures["serving_default"]
                            predict = infer(tf.convert_to_tensor(merge_img, dtype = tf.float32))
                            #predict = model.predict(merge_img)
                            predict_tensor = predict['dense_1']
                            predict_np = predict_tensor.numpy()
                            #print(predict_np.shape)
                            #print(predict_np)
                            yhat = np.argmax(predict_np)
                            #print(yhat)

                            if yhat == 0:
                                print("1")
                                cv2.putText(frame, "1", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_1x1 = str(round((predict_np[0][0]), 4)) +'%'
                                cv2.putText(frame,str_1x1 , (crop_fist_x + 115, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 10
                            elif yhat == 1:
                                print("10")
                                cv2.putText(frame, "10", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_1x4 = str(round((predict_np[0][1]), 4)) +'%'
                                cv2.putText(frame,str_1x4 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 30
                            elif yhat == 2:
                                print("11")
                                cv2.putText(frame, "11", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][2]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 50
                            elif yhat == 3:
                                print("2")
                                cv2.putText(frame, "2", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][3]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 70
                            elif yhat == 4:
                                print("3")
                                cv2.putText(frame, "3", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][4]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 90
                            elif yhat == 5:
                                print("4")
                                cv2.putText(frame, "4", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][5]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 110
                            elif yhat == 6:
                                print("5")
                                cv2.putText(frame, "5", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][6]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 130
                            elif yhat == 7:
                                print("6")
                                cv2.putText(frame, "6", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][7]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 145
                            elif yhat == 8:
                                print("7")
                                cv2.putText(frame, "7", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][8]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 160
                            elif yhat == 9:
                                print("8")
                                cv2.putText(frame, "8", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][9]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 175
                            elif yhat == 10:
                                print("9")
                                cv2.putText(frame, "9", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                                str_2x6 = str(round((predict_np[0][10]), 4)) +'%'
                                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
                                msg['sorting_step'] = 180
                            else:
                                print("base")
                            
                            json_msg = json.dumps(msg)
                            print(json_msg)
                            self.ser.write(json_msg.encode())
                            self.ser.write(end_str.encode())
                
                

                img_list[1] = morph_close
                self.morph_flag = True
                
            terminate_t = timeit.default_timer()
            FPS = int(1. / (terminate_t - start_t))
            fps_str = "FPS : %0.1f" % FPS
            cv2.putText(frame, fps_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))  
            img_list[0] = frame
            QApplication.processEvents()         
            lock.release()



        cap.release()
        self.run_flag = False
        print("Thread end.")

    def stop(self):
        global running
        running = False
        print("stoped..")

    def start(self):
        global running
        running = True
        th = threading.Thread(target=self.video_run)
        th.daemon = True
        th.start()
        print("started..")

    def onExit(self):
        print("exit")
        stop()

    def change_mode(self, state):
        if state == Qt.Checked:
            self.start()
        else:
            self.stop()
        print(state)
    
    def serial_connect(self):
        self.logwindow.append("Serial Connect...")
        self.serial_btn.toggle()

        try:
            self.ser = serial.Serial(port,baud)
            self.logwindow.append("Ok!!")
        except:
            self.logwindow.append("Failed...")

    def model_load(self):
        print("model_load_thread start!!!")
    
    def opencv_apply(self, state):
        if state == Qt.Checked:
            self.opencv_apply_flag = True
            self.logwindow.append("Opencv apply On!")
        else:
            self.opencv_apply_flag = False
            self.logwindow.append("Opencv apply Off!")
    
    def Dnn_infer_apply(self, state):
        if state == Qt.Checked:
            self.Dnn_infer_apply_flag = True
            self.logwindow.append("Dnn_infer apply On!")
        else:
            self.Dnn_infer_apply_flag = False
            self.logwindow.append("Dnn_infer apply Off!")

    def draw_ui(self):
        if len(img_list) > 1:
   
            if self.run_flag  == True:
                frame = img_list[0]    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                h,w,c = frame.shape
                qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.image_label.setPixmap(pixmap)

            if self.morph_flag == True:
                morph_close = img_list[1]
                morph_close = cv2.cvtColor(morph_close, cv2.COLOR_BGR2RGB) 
                morph_h,morph_w,morph_c = morph_close.shape
                morph_qImg = QtGui.QImage(morph_close.data, morph_w, morph_h, morph_w*morph_c, QtGui.QImage.Format_RGB888)
                morph_pixmap = QtGui.QPixmap.fromImage(morph_qImg)
                self.edge_label.setPixmap(morph_pixmap)
            
            if self.opencv_apply_flag == True:
                for i in range(0, len(memory)):
                    detect_img = cv2.cvtColor(memory[i], cv2.COLOR_BGR2RGB) 
                    h,w,c = detect_img.shape
                    qImg = QtGui.QImage(detect_img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    if i == 0:
                        self.detect_label0.setPixmap(pixmap)
                    elif i == 1:
                        self.detect_label1.setPixmap(pixmap)
                    elif i == 2:
                        self.detect_label2.setPixmap(pixmap)
                    elif i == 3:
                        self.detect_label3.setPixmap(pixmap)
                    elif i == 4:
                        self.detect_label4.setPixmap(pixmap)
                    elif i == 5:
                        self.detect_label5.setPixmap(pixmap)
   
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    Timer = QTimer()
    Timer.setInterval(1)
    Timer.timeout.connect(lambda:ex.draw_ui())
    Timer.start()
    sys.exit(app.exec_())



