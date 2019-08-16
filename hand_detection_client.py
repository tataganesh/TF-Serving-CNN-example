from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import cv2
import numpy as np
import tensorflow as tf
from tf_serving import serving_config


class handsDetection:
    def __init__(self, host, port, model_name, signature_name="serving_default"):
        self.channel = grpc.insecure_channel(host + ":" + str(port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = signature_name
    def predict(self, image, confidence):
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(rgb_image, axis=0)
        self.request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape))
        result = self.stub.Predict(self.request, 10.0)
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
        return hand_boxes






