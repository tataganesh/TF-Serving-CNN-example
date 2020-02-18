# Serving a Tensorflow CNN model for Hand Detection using [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)


## DataHack Summit 2019 Slides
I conducted a hack session at DataHack Summit 2019 on [Model Deployment using Tensorflow Serving](https://www.analyticsvidhya.com/datahack-summit-2019/schedule/hack-session-all-you-need-to-know-about-deploying-dl-models-using-tensorflow/). Here are the slides - [Talk slides](https://docs.google.com/presentation/d/1-NKPK4XU8BXYBbre_GYseep4RAwPl0ebzscIgOb00bw/edit?usp=sharing)


## Model Training

I referred to the following work for training the hand detector model - 

Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks, https://github.com/victordibia/handtracking 

## File and folder Descriptions

* [ball_with_hand.py](ball_with_hand.py) - Driver code to run the hand detection model on video frames
* [hand_detection_client.py](hand_detection_client.py) - Classes that enable interation with the model using gRPC or REST
* [serving_config.py](tf_serving/serving_config.py) - Port and host configuration to interact with the served model
* [inference_graph](inference_graph) - Saved model for Hand Detection

## Dependencies

* [python dependencies](requirements.txt)
* [Install Tensorflow Serving using Docker](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md)

## Example

* Download the [sample video](https://drive.google.com/file/d/1W9Mi51EICjUCk0HPDrV2qnV6CMMc62Ra/view?usp=sharing)
* Serve the hand detector

    ```bash
    docker run -p 8501:8500 -p 9000:8501  \
        --mount type=bind,source=/home/tata/Projects/cnn-hand-detector/inference_graph/,target=/models/inference_graph  \ 
        --mount type=bind,source=/home/tata/Projects/cnn-hand-detector/inference_graph/model_config.config,target=/models/model_config.config \  
        -t -i tensorflow/serving --model_config_file=/models/model_config.config --model_config_file_poll_wait_seconds=10 \  --enable_model_warmup=true
    ```
    [![asciicast](https://asciinema.org/a/L86BFXVcRqCngY6Y60UPI2I12.svg)](https://asciinema.org/a/L86BFXVcRqCngY6Y60UPI2I12)
    The server can now accept gRPC requests through port 8501 and REST requests through port 9000.
* Run [ball_with_hand.py](ball_with_hand.py) ( change the path of `vid` to the downloaded video's path).

## More details will be added soon to this README. 