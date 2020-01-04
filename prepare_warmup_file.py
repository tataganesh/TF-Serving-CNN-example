import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_log_pb2
from tf_serving import serving_config
import grpc
import cv2
import numpy as np

def main():
    with tf.python_io.TFRecordWriter("/home/tata/Projects/hand_detector/inference_graph/1/assets.extra/tf_serving_warmup_requests") as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = serving_config.model_name
        image = cv2.imread('/home/tata/hand2.jpg')
        image_batch = np.array([image] * 5) # Use a batch of 5 images for warmup
        request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_batch, shape=image_batch.shape))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())

if __name__ == "__main__":
    main()