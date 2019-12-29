from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import cv2
import numpy as np
import tensorflow as tf
from tf_serving import serving_config
import requests
import json
import IPython
import time


class handsDetection:
    def __init__(self, host, port, model_name, signature_name="serving_default"):
        self.channel = grpc.insecure_channel(host + ":" + str(port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = signature_name
        self.count = 0
        self.inputPtime = 0
        self.reqResTime = 0
    def predict(self, image, confidence):
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
        self.count += 1 
        # if self.count % 5 == 0:
            # self.request.model_spec.version_label = "canary"
        # else:
            # self.request.model_spec.version_label = "stable"
        image_np_expanded = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation     
        self.request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape))
        self.inputPtime = time.time() - start
        result = self.stub.Predict(self.request, 10.0)
        end = time.time()
        self.reqResTime += end - start
        time_taken = end - start
        outputs = result.outputs
        detection_classes = tf.make_ndarray(outputs["detection_classes"])
        detection_boxes = tf.make_ndarray(outputs["detection_boxes"])
        detection_scores = tf.make_ndarray(outputs["detection_scores"])
        detection_boxes, detection_scores, detection_classes = map(
            np.squeeze, [detection_boxes, detection_scores, detection_classes])
        hand_boxes = list()
        for box, score in zip(detection_boxes, detection_scores):
            if(score > confidence):
                y1,x1,y2,x2 = box[0]*image.shape[0], box[1]*image.shape[1], box[2]*image.shape[0], box[3]*image.shape[1]
                x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])
                hand_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return hand_boxes, time_taken

class handsDetectionRest:
    def __init__(self, host, port, model_name, signature_name="serving_default"):
        self.urlPrefix = "http://" + host + ":" + str(port) + "/v1/models/" +  model_name  + "/versions/"
        self.signature_name = signature_name
        self.headers = {"content-type": "application/json"}
        self.count = 0
        self.inputPtime = 0
        self.reqResTime = 0
    def predict(self, image, confidence, version_number=1):
        self.count += 1
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (300, 300))
        image_np_expanded = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation 
        data = json.dumps({"signature_name": self.signature_name, "inputs": {"inputs": image_np_expanded.tolist()}})
        self.inputPtime = time.time() - start
        url = self.urlPrefix + str(version_number) + ":predict"
        json_response = requests.post(url, data=data, headers=self.headers)
        end = time.time()
        self.reqResTime += (end - start) 
        time_taken = end - start
        result = json.loads(json_response.text)
        output = result['outputs']
        detection_classes = output["detection_classes"]
        detection_boxes = output["detection_boxes"]
        detection_scores = output["detection_scores"]
        detection_boxes, detection_scores, detection_classes = map(
            np.squeeze, [detection_boxes, detection_scores, detection_classes])
        hand_boxes = list()
        for box, score in zip(detection_boxes, detection_scores):
            if(score > confidence):
                y1,x1,y2,x2 = box[0]*image.shape[0], box[1]*image.shape[1], box[2]*image.shape[0], box[3]*image.shape[1]
                x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])
                hand_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return hand_boxes, time_taken







