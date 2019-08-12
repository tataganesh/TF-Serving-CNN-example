import tensorflow as tf
import cv2
import numpy as np
import os
import IPython
PATH_TO_FROZEN_GRAPH = '/home/tata/Projects/hand_detector/inference_graph/frozen_inference_graph.pb'


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def detect(image, sess):

    # image = cv2.imread(image_path)
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
    img = image
    for box, score in zip(boxes, scores):
        print(box, score)
        if(score > 0.05):
            # y1, x1, y2, x2 = 
            y1,x1,y2,x2 = box[0]*img.shape[0], box[1]*img.shape[1], box[2]*img.shape[0], box[3]*img.shape[1]
            cv2.rectangle(image, (int(x1), int(y1)), ((int(x2), int(y2))), (255, 0, 0), 2)
    return image

session = tf.Session(graph=detection_graph)
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = detect(img, session)
        cv2.imshow('my webcam', cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
    show_webcam()