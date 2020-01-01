from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import cv2
import numpy as np
import tensorflow as tf
from tf_serving import serving_config
import requests
import json
import time
CONFIDENCE = 0.1
RESIZE_WIDTH = 300
RESIZE_HEIGHT = 300

def extractBboxes(networkOutput, imageHeight, imageWidth):
    detection_boxes, detection_scores, detection_classes = map(
    np.squeeze, networkOutput)
    hand_boxes = list()
    for box, score in zip(detection_boxes, detection_scores):
        if(score > CONFIDENCE):
            y1, x1, y2, x2 = box[0]*imageHeight, box[1]*imageWidth, box[2]*imageHeight, box[3]*imageWidth
            hand_box = map(int, [x1, y1, x2, y2])
            hand_boxes.append(tuple(hand_box))
    return hand_boxes
    
class handsDetection:
    def __init__(self, host, port, model_name, signature_name="serving_default"):
        self.channel = grpc.insecure_channel(host + ":" + str(port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = signature_name
        self.count = 0
        self.inputProcessingTime = 0
        self.reqResTime = 0

    def predict(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        self.count += 1 
        # if self.count % 5 == 0:
            # self.request.model_spec.version_label = "canary"
        # else:
            # self.request.model_spec.version_label = "stable"
        image_np_expanded = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation     
        self.request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape))
        self.inputProcessingTime = time.time() - start
        # Prediction
        result = self.stub.Predict(self.request, 10.0)
        end = time.time()
        self.reqResTime += end - start
        time_taken = end - start
        outputs = result.outputs
        detection_classes = tf.make_ndarray(outputs["detection_classes"])
        detection_boxes = tf.make_ndarray(outputs["detection_boxes"])
        detection_scores = tf.make_ndarray(outputs["detection_scores"])
        hand_boxes = extractBboxes([detection_boxes, detection_scores, detection_classes], height, width)
        return hand_boxes, time_taken

class handsDetectionRest:
    def __init__(self, host, port, model_name, signature_name="serving_default"):
        self.urlPrefix = "http://" + host + ":" + str(port) + "/v1/models/" +  model_name  + "/versions/"
        self.signature_name = signature_name
        self.headers = {"content-type": "application/json"}
        self.count = 0
        self.inputProcessingTime = 0
        self.reqResTime = 0

    def predict(self, image, version_number=1):
        self.count += 1
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        rgb_image = cv2.resize(rgb_image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image_np_expanded = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation 
        data = json.dumps({"signature_name": self.signature_name, "inputs": {"inputs": image_np_expanded.tolist()}})
        self.inputProcessingTime = time.time() - start
        url = self.urlPrefix + str(version_number) + ":predict"
        # predict
        response = requests.post(url, data=data, headers=self.headers)
        end = time.time()
        self.reqResTime += (end - start) 
        time_taken = end - start
        result = json.loads(response.text)
        output = result['outputs']
        detection_classes = output["detection_classes"]
        detection_boxes = output["detection_boxes"]
        detection_scores = output["detection_scores"]
        hand_boxes = extractBboxes([detection_boxes, detection_scores, detection_classes], height, width)
        return hand_boxes, time_taken







