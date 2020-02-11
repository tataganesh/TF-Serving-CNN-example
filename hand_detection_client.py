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

def extract_bboxes(network_output, image_height, image_width):   
    """Extract bounding boxes from output of Hand Detection network
    
    Args:
        network_output (ndarray): Output of Hand detection network
        image_height (int): Image width
        image_width (int): Image height
    Returns:
        list: List of tuples. Each tuple represents a Hand box's coordinates - (x1, y1, x2, y2)
    """        
    detection_boxes, detection_scores, _ = map(
    np.squeeze, network_output)
    hand_boxes = list()
    for box, score in zip(detection_boxes, detection_scores):
        if(score > CONFIDENCE):
            y1, x1, y2, x2 = box[0]*image_height, box[1]*image_width, box[2]*image_height, box[3]*image_width
            hand_box = map(int, [x1, y1, x2, y2])
            hand_boxes.append(tuple(hand_box))
    return hand_boxes
    
class handsDetection:
    """Class to extract hand boxes from images. Interaction of the model is through gRPC."""

    def __init__(self, host, port, model_name, signature_name="serving_default"):
        """init method
        
        Args:
            host (string): Hostname of the server
            port (int): gRPC port to interact with the served model
            model_name (string): Name of the served model 
            signature_name (str, optional): Model signature name. Defaults to "serving_default".
        """        
        self.channel = grpc.insecure_channel(host + ":" + str(port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = signature_name
        self.count = 0
        self.input_processing_time = 0
        self.req_res_time = 0

    def predict(self, image):
        """Function to get hand boxes from an image
        
        Args:
            image (ndarray): Input image
        
        Returns:
            tuple: The tuple contains two elements.
                    1. hand boxes 
                    2. Avg. Time taken for inferencing
        """        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        self.count += 1 
        # if self.count % 5 == 0:
            # self.request.model_spec.version_label = "canary"
        # else:
            # self.request.model_spec.version_label = "stable"
        image_batch = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation     
        self.request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_batch, shape=image_batch.shape))
        self.input_processing_time = time.time() - start
        # Prediction
        result = self.stub.Predict(self.request, 10.0)
        end = time.time()
        self.req_res_time += end - start
        time_taken = end - start
        outputs = result.outputs
        detection_classes = tf.make_ndarray(outputs["detection_classes"])
        detection_boxes = tf.make_ndarray(outputs["detection_boxes"])
        detection_scores = tf.make_ndarray(outputs["detection_scores"])
        hand_boxes = extract_bboxes([detection_boxes, detection_scores, detection_classes], height, width)
        return hand_boxes, time_taken

class handsDetectionRest:
    """Class to extract hand boxes from images. Interaction of the model is through REST."""

    def __init__(self, host, port, model_name, signature_name="serving_default"):
        """init method
        
        Args:
            host (string): Hostname of the server
            port (int): gRPC port to interact with the served model
            model_name (string): Name of the served model 
            signature_name (str, optional): Model signature name. Defaults to "serving_default".
        """         
        self.url_prefix = "http://" + host + ":" + str(port) + "/v1/models/" +  model_name  + "/versions/"
        self.signature_name = signature_name
        self.headers = {"content-type": "application/json"}
        self.count = 0
        self.input_processing_time = 0
        self.req_res_time = 0

    def predict(self, image, version_number=1):
        """Function to get hand boxes from an image
        
        Args:
            image (ndarray): Input image
            version_number (int, optional): Specific version of the model. Defaults to 1.
        
        Returns:
            tuple: The tuple contains two elements.
                    1. hand boxes 
                    2. Avg. Time taken for inferencing
        """        
        self.count += 1
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image_batch = np.expand_dims(rgb_image, axis=0) # Sending a batch of 1
        start = time.time()
        # Input preparation 
        data = json.dumps({"signature_name": self.signature_name, "inputs": {"inputs": image_batch.tolist()}})
        self.input_processing_time = time.time() - start
        url = self.url_prefix + str(version_number) + ":predict"
        # predict
        response = requests.post(url, data=data, headers=self.headers)
        end = time.time()
        self.req_res_time += (end - start) 
        time_taken = end - start
        result = json.loads(response.text)
        output = result['outputs']
        detection_classes = output["detection_classes"]
        detection_boxes = output["detection_boxes"]
        detection_scores = output["detection_scores"]
        height, width = rgb_image.shape[:2]
        hand_boxes = extract_bboxes([detection_boxes, detection_scores, detection_classes], height, width)
        return hand_boxes, time_taken







