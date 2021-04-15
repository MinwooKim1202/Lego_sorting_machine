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
        cap = cv2.VideoCapture('test.mp4')
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

model = load_model('lego_sorter.h5')
cap, mog2 = video_read()
# BackgroundSubtractorMOG2
# cap = cv2.VideoCapture('test4.mp4')
# mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
i = 0

while True:
    ret, frame = cap.read()

    start_t = timeit.default_timer()

    frame = imutils.resize(frame, width=720)
    frame = imutils.rotate(frame, angle=-90)
    frame = frame[0:400, 160:560]
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

    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        margin = 10
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        if w > 70 and h > 70:
            crop_fist_x = clamp(x - margin)
            crop_fist_y = clamp(y - margin)
            crop_twice_x = clamp(x + w + margin)
            crop_twice_y = clamp(y + h + margin)
            cropped_img = origin_frame[crop_fist_y: crop_twice_y, crop_fist_x: crop_twice_x]
            cropped_img = cv2.resize(cropped_img, dsize=(197, 197), interpolation=cv2.INTER_LINEAR)
            #print(cropped_img.shape)
            img_b, img_g, img_r = cv2.split(cropped_img)
            img_b = img_b / 255
            img_g = img_g / 255
            img_r = img_r / 255
            merge_img = cv2.merge((img_r, img_g, img_b))
            merge_img = merge_img.reshape((1, 197, 197, 3))
            print(model.predict(merge_img))
            yhat = np.argmax(model.predict(merge_img))
            #print(yhat)
            if yhat == 0:
                print("1x1_brick")
            elif yhat == 1:
                print("1x4_brick")
            elif yhat == 2:
                print("2x6_brick")
            else:
                print("base")
            #file_name = 'cropped_img/img' + str(i) + '.jpg'
            cv2.imshow('cropped_img', cropped_img)
            #cv2.imwrite(file_name, cropped_img)
            #i = i + 1

    terminate_t = timeit.default_timer()
    FPS = int(1. / (terminate_t - start_t))
    fps_str = "FPS : %0.1f" % FPS
    cv2.putText(frame, fps_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow('edge', edge)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()