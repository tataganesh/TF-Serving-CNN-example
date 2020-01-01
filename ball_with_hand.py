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

if args.method == "rest": 
    hands_detect_model = hand_detection_client.handsDetectionRest(serving_config.host, serving_config.restAPIport, serving_config.model_name)
else:   
    hands_detect_model = hand_detection_client.handsDetection(serving_config.host, serving_config.port, serving_config.model_name)


def run_on_video(vid_path=None):
    if vid_path is not None:
        cam = cv2.VideoCapture(vid_path)
    else:
        cam = cv2.VideoCapture(0)
    while True:
        ret_val, bgrImage = cam.read()
        if not ret_val:
            break
        ball_box = helpers.find_ball(bgrImage)
        hand_boxes, time_taken = hands_detect_model.predict(bgrImage)
        if hand_boxes:
            bgrImage = helpers.draw_box_with_ball(hand_boxes, ball_box, bgrImage)
        cv2.putText(bgrImage,'%.3f'%(time_taken), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('my webcam', bgrImage)
        if args.hold:
            cv2.waitKey(0)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

vid = '/home/tata/hand_video.mp4'
run_on_video(vid)