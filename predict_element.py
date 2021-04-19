import cv2
import numpy
import matplotlib
import imutils
import timeit
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model


def clamp(val):
    if val < 0:
        val = 0
    return val


def backSubtraction(frame):
    fgmask = mog2.apply(frame)
    return fgmask


def saturare(frame):
    saturate_val = 70
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(imghsv)
    s = s + saturate_val
    s = numpy.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgrgb


def video_read():
    try:
        saturate = 100
        print('비디오를 읽어옵니다.')
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture('video/v3_test2.mp4')
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=720)
        # frame = imutils.rotate(frame, angle=-90)
        frame = frame[0:400, 160:560]

        mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # cv2.imshow('saturate', imgrgb)

        return cap, mog2
    except:
        print('실패')
        return

model = load_model('lego_sorter_v3.h5')
cap, mog2 = video_read()
# BackgroundSubtractorMOG2
# cap = cv2.VideoCapture('test4.mp4')
# mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
video_width = int(cap.get(3))
video_height = int(cap.get(4))
out = cv2.VideoWriter('save_video/demo1.avi', fcc, 30, (video_width, video_height))

i = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    start_t = timeit.default_timer()
    if not ret:
        print("영상 종료")
        break

    video_fps = cv2.CAP_PROP_FPS

    frame = imutils.resize(frame, width=720)
    #frame = imutils.rotate(frame, angle=-90)
    #frame = frame[0:400, 160:560]
    # frame = saturare(frame)
    origin_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)  # Opening
    edge = cv2.Canny(morph, 50, 200)
    # fgmask = mog2.apply(morph)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)  # Opening
    # fgmask = backSubtraction(morph) # MOG2

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        margin = 10
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        if w > h:  # 물체가 rect 중앙에 오게함
            y = int(y - ((w - h) / 2))
            h = int(h + (((w - h) / 2) * 2))
        else:
            x = int(x - ((h - w) / 2))
            w = int(w + (((h - w) / 2) * 2))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        if w > 30 and h > 30:
            crop_fist_x = clamp(x - margin)
            crop_fist_y = clamp(y - margin)
            crop_twice_x = clamp(x + w + margin)
            crop_twice_y = clamp(y + h + margin)
            cropped_img = origin_frame[crop_fist_y: crop_twice_y, crop_fist_x: crop_twice_x]
            cropped_img = cv2.resize(cropped_img, dsize=(88, 88), interpolation=cv2.INTER_LINEAR)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            if abs(cropped_img.shape[0] - cropped_img.shape[1]) > 10: # width 와 height 가 10이상 차이나면 저장 x
                continue
            #img_b, img_g, img_r = cv2.split(cropped_img)
            #img_b = img_b / 255
            #img_g = img_g / 255
            #img_r = img_r / 255
            #merge_img = cv2.merge((img_r, img_g, img_b))

            cropped_img = cropped_img / 255
            merge_img = cropped_img.reshape((1, 88, 88, 1))
            predict = model.predict(merge_img)
            yhat = np.argmax(predict)
            #print(yhat)
            if yhat == 0:
                print("1x1_brick")
                cv2.putText(frame, "1x1_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_1x1 = str(round((predict[0][0]), 4)) +'%'
                cv2.putText(frame,str_1x1 , (crop_fist_x + 115, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 1:
                print("1x2_brick")
                cv2.putText(frame, "1x2_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_1x4 = str(round((predict[0][1]), 4)) +'%'
                cv2.putText(frame,str_1x4 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 2:
                print("1x3_brick")
                cv2.putText(frame, "1x3_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_2x6 = str(round((predict[0][2]), 4)) +'%'
                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 3:
                print("1x4_brick")
                cv2.putText(frame, "1x4_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_2x6 = str(round((predict[0][3]), 4)) +'%'
                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 4:
                print("2x2_brick")
                cv2.putText(frame, "2x2_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_2x6 = str(round((predict[0][4]), 4)) +'%'
                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 5:
                print("2x3_brick")
                cv2.putText(frame, "2x3_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_2x6 = str(round((predict[0][5]), 4)) +'%'
                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            elif yhat == 6:
                print("2x6_brick")
                cv2.putText(frame, "2x6_brick", (crop_fist_x, crop_fist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 0, 153))
                str_2x6 = str(round((predict[0][6]), 4)) +'%'
                cv2.putText(frame,str_2x6 , (crop_fist_x + 115, crop_fist_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
            else:
                print("base")
            #file_name = 'cropped_img/img' + str(i) + '.jpg'
            cv2.imshow('cropped_img', cropped_img)
            #cv2.imwrite(file_name, cropped_img)
            #i = i + 1
    #out.write(frame)
    terminate_t = timeit.default_timer()
    FPS = int(1. / (terminate_t - start_t))
    fps_str = "FPS : %0.1f" % FPS
    cv2.putText(frame, fps_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
    cv2.imshow('edge', edge)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()