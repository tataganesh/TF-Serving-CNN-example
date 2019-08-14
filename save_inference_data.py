import tensorflow as tf
import cv2
import numpy as np
import IPython
import os
import pandas as pd
PATH_TO_FROZEN_GRAPH = '/home/tata/Downloads/frozen_inference_graph.pb'
TEST_IMAGES_PATH = '/home/tata/Projects/hand_detector/hands_dataset/test'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def detect(image_path, sess):
    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    image_np_expanded = np.expand_dims(image, axis=0)
    detections = list()
    graph = detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    img = image
    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])
    for box, score in zip(boxes, scores):
        if(score > 0.02):
            # y1,x1,y2,x2 = box[0]*img.shape[0], box[1]*img.shape[1], box[2]*img.shape[0], box[3]*img.shape[1]
            y1, x1, y2, x2 = box
            detections.append([image_name, "hand", x1, y1, x2, y2, height, width])
    # return image

    return detections

session = tf.Session(graph=detection_graph)
df = pd.DataFrame(columns=['image_name', 'class_name', 'x1', 'y1', 'x2', 'y2', 'height', 'width'])

for image_path in os.listdir(TEST_IMAGES_PATH):
    detections = detect(os.path.join(TEST_IMAGES_PATH, image_path), session)
    for detection in detections:
        df.loc[len(df)] = detection


df.to_csv("test_inference_data.csv", index=False)