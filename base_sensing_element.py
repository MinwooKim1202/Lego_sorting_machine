import cv2
import numpy
import matplotlib
import imutils
import timeit

def clamp(val):
    if val < 0:
        val = 0
    return val

def backSubtraction(frame):
    fgmask = mog2.apply(frame)
    return fgmask

def saturare(frame):
    saturate_val = 30
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
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture('base.mp4')
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=720)
        #frame = imutils.rotate(frame, angle=-90)
        frame = frame[0:400, 150:560]

        mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        #cv2.imshow('saturate', imgrgb)
     
        return cap, mog2
    except:
        print('실패')
        return

cap, mog2 = video_read()
# BackgroundSubtractorMOG2
#cap = cv2.VideoCapture('test4.mp4')
#mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
i = 0


while True:
    ret, frame = cap.read()

    start_t = timeit.default_timer()

    frame = imutils.resize(frame, width=720)
    frame = imutils.rotate(frame, angle=-90)
    frame = frame[0:400, 150:560]
    origin_frame = frame.copy()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel) # Opening
    edge = cv2.Canny(morph, 50, 200)
    #fgmask = mog2.apply(morph)
    #edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel) # Opening
    #fgmask = backSubtraction(morph) # MOG2 

    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cropped_img = frame[100:400, 100:400]
    file_name = 'cropped_img/img' + str(i) + '.jpg'
    cv2.imshow('cropped_img', cropped_img)
    cv2.imwrite(file_name, cropped_img)

    i = i + 1

    
    terminate_t = timeit.default_timer()
    FPS = int(1./(terminate_t - start_t ))
    fps_str = "FPS : %0.1f" % FPS
    cv2.putText(frame, fps_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow('video', edge)
    cv2.imshow('origin_video', origin_frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()