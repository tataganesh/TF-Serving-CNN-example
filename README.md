# Serving a Tensorflow CNN model for Hand Detection using [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)


## DataHack Summit 2019 Slides
I conducted a hack session at DataHack Summit 2019 on [Model Deployment using Tensorflow Serving](https://www.analyticsvidhya.com/datahack-summit-2019/schedule/hack-session-all-you-need-to-know-about-deploying-dl-models-using-tensorflow/). Here are the slides - [Talk slides](https://docs.google.com/presentation/d/1-NKPK4XU8BXYBbre_GYseep4RAwPl0ebzscIgOb00bw/edit?usp=sharing)


## Model Training

I referred to the following work for training the hand detector model - 

Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks, https://github.com/victordibia/handtracking 

## File and folder Descriptions

* [ball_with_hand.py](https://github.com/ganeshtata/TF-Serving-CNN-example/blob/add_readme/ball_with_hand.py) - Driver code to run the hand detection model on video frames
* [hand_detection_client.py](https://github.com/ganeshtata/TF-Serving-CNN-example/blob/add_readme/hand_detection_client.py) - Classes that enable interation with the model using gRPC or REST
* [serving_config.py](https://github.com/ganeshtata/TF-Serving-CNN-example/blob/add_readme/tf_serving/serving_config.py) - Port and host configuration to interact with the served model
* [hand_inference_graph](https://github.com/ganeshtata/TF-Serving-CNN-example/tree/add_readme/hand_inference_graph) - Saved model for Hand Detection


## More details will be added soon to this README. 