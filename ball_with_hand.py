import tensorflow as tf
import cv2
import numpy as np
import os
import helpers
import hand_detection_client
from tf_serving import serving_config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hold", help="Wait for key press for every image", action='store_true')
parser.add_argument("--method", help="Grpc / Rest", default="grpc")
args = parser.parse_args()

# version_label_func = lambda x: "stable" if x%5 else "canary"
version_label_func = None
if args.method == "rest": 
    hands_detect_model = hand_detection_client.handsDetectionRest(serving_config.host, serving_config.rest_api_port, serving_config.model_name)
else:   
    hands_detect_model = hand_detection_client.handsDetection(serving_config.host, serving_config.port, serving_config.model_name, version_label_func=version_label_func)

def run(vid_path=None):
    if vid_path is not None:
        cam = cv2.VideoCapture(vid_path)
    else:
        cam = cv2.VideoCapture(0)
    while True:
        ret_val, bgr_image = cam.read()
        if not ret_val:
            break
        ball_box = helpers.find_green_ball(bgr_image)
        hand_boxes, time_taken = hands_detect_model.predict(bgr_image)
        if hand_boxes:
            bgr_image = helpers.draw_box_with_ball(hand_boxes, ball_box, bgr_image)
        cv2.putText(bgr_image,'%.3f'%(time_taken), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Spot the ball', bgr_image)
        if args.hold:
            cv2.waitKey(0)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

vid = '/home/tata/hand_video.mp4'
run(vid)