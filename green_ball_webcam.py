import cv2
from detect_colour import find_ball
cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    rect = find_ball(img)
    if rect:
        cv2.rectangle(img, rect[0], rect[1], (0,255,0), 3)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
cam.release()