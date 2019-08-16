import tensorflow as tf
import cv2
import numpy as np
import os
import IPython
import helpers
import hand_detection_client
from tf_serving import serving_config



def draw_box_with_ball(boxes, ball_box, image):
    box_colour_mapping = dict()
    if not ball_box:
        for box in boxes:
            box_colour_mapping[box] = (0, 0, 255)
    else:
        for box in boxes:
            if helpers.bb_intersection_over_union(box, ball_box) > 0.5:
                box_colour_mapping[box] = (0, 255, 0)
            else:
                box_colour_mapping[box] = (0, 0, 255)
    for box, colour in box_colour_mapping.items():
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour, 3)
    return image


hands_detect_model = hand_detection_client.handsDetection(serving_config.host, serving_config.port, serving_config.model_name)
def show_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, bgrImage = cam.read()
        ball_box = helpers.find_ball(bgrImage)
        hand_boxes = hands_detect_model.predict(bgrImage, confidence=0.1)
        if hand_boxes:
            bgrImage = draw_box_with_ball(hand_boxes, ball_box,bgrImage)
        cv2.imshow('my webcam', bgrImage)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
show_webcam()

