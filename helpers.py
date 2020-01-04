import cv2
import numpy as np
import time

lower_green = np.array([15, 50, 50])
upper_green = np.array([90, 255, 255])

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

def draw_box_with_ball(boxes, ball_box, image):
    box_colour_mapping = dict()
    original_image = image.copy()
    blurred_image = cv2.GaussianBlur(image,(21, 21), 0)
    if not ball_box:
        for box in boxes:
            box_colour_mapping[box] = (0, 0, 255)
    else:
        # cv2.rectangle(image, ball_box[0:2], ball_box[2:4], (255,0,0), 3)
        for box in boxes:
            if bb_intersection_over_union(box, ball_box) > 0.5:
                box_colour_mapping[box] = (0, 255, 0)
            else:
                box_colour_mapping[box] = (0, 0, 255)
    for box, colour in box_colour_mapping.items():
        blurred_image[box[1]: box[3], box[0]: box[2]] = original_image[box[1]: box[3], box[0]: box[2]]
        cv2.rectangle(blurred_image, (box[0], box[1]), (box[2], box[3]), colour, 3)
    return blurred_image

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interAreaRatio = max(0, xB - xA + 1) * max(0, yB - yA + 1) * 1.0/ ((boxB[3] - boxB[1]) * (boxB[2] - boxB[0])) * 1.0
	return interAreaRatio

def find_ball(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    res = cv2.bitwise_and(img, hsv_image, mask=mask)
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    retval, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    contours, heirarchy = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=lambda k:cv2.contourArea(k))
        if not cv2.contourArea(largest_contour) < 100: 
            empty = np.zeros_like(gray_image)
            x,y,w,h = cv2.boundingRect(largest_contour)
            return (x - 15, y - 15, x+w + 20, y + h + 20)