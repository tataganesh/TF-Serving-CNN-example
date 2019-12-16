import tensorflow as tf
import cv2
import numpy as np
import os
import IPython
import helpers
PATH_TO_FROZEN_GRAPH = '/home/tata/Projects/cnn-hand-detector/hand_inference_graph/frozen_inference_graph.pb'


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def detect(image, sess):

    image_np_expanded = np.expand_dims(image, axis=0)
    height, width = image.shape[:2]
    graph = detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])
    hand_boxes = list()
    for box, score in zip(boxes, scores):
        if(score > 0.02):
            y1,x1,y2,x2 = box[0]*image.shape[0], box[1]*image.shape[1], box[2]*image.shape[0], box[3]*image.shape[1]
            hand_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return hand_boxes

session = tf.Session(graph=detection_graph)


def draw_box_with_ball(boxes, ball_box, image):
    box_colour_mapping = dict()
    if not ball_box:
        for box in boxes:
            box_colour_mapping[box] = (0, 0, 255)
    else:
        # cv2.rectangle(image, ball_box[0:2], ball_box[2:4], (255,0,0), 3)
        for box in boxes:
            if helpers.bb_intersection_over_union(box, ball_box) > 0.01:
                box_colour_mapping[box] = (0, 255, 0)
            else:
                box_colour_mapping[box] = (0, 0, 255)
    for box, colour in box_colour_mapping.items():
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour, 3)
    return image


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, bgrImage = cam.read()
        ball_box = helpers.find_ball(bgrImage)
        rgbImage = cv2.cvtColor(bgrImage,cv2.COLOR_BGR2RGB)
        hand_boxes = detect(rgbImage, session)
        if hand_boxes:
            bgrImage = draw_box_with_ball(hand_boxes, ball_box,bgrImage)
        cv2.imshow('my webcam', bgrImage)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    

if __name__ ==  "__main__":
    show_webcam()