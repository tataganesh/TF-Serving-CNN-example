xfrom camera_hand_detection import detect
import tensorflow as tf
import cv2
import numpy as np

vid = '/home/tata/video.mp4'
PATH_TO_FROZEN_GRAPH = '/home/tata/Projects/hand_detector/output_inference_graph/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

session = tf.Session(graph=detection_graph)
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(vid)
    while cam.isOpened():
        ret_val, img = cam.read()
        print(ret_val)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = detect(img, session)
        cv2.imshow('my webcam', img)
        cv2.waitKey(0)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

show_webcam()