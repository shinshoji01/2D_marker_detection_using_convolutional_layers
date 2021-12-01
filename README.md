# Marker Detection
This is to introduce my solution to one of the assignments; the algorithm for marker detection.

---
## Introduction
I've conducted 2D-images marker detection with [ArUco markers](https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html). Since the detection model is made up of a convolutional layer and pooling layer, I strongly recommend using GPU to save your time. The detection process can be divided into 3 steps; frame detection, extraction, and comparison. The sample is shown below.


<img src="./data/images/representation.png" width="500">

---
## Procedure
As I said, the process segments into 3 parts: frame detection, extraction, and comparison.

### frame detection
This process is to eliminate most of the pixels that are irrelevalnt to the center position of the markers. All the samples of the marker have a black frame as you can see from the previous example, so, at first, the model detects the position of the frames, and also the rotation and the size of the markers by using various size and angle of frame activation filters, which are constructed by a convolutional layer. The samples of the filter are illustrated below.

<img src="./data/images/frame.png" width="100"><img src="./data/images/frame_rotated.png" width="100">

After this process, you can get the score of activation in every location with regard to various sizes and rotations. When you compute the maximum value referring to various sizes and rotations, you can get the scores as shown below.

<img src="./data/images/miximimized_location.png" width="500">

Even though numerous pixels are activated around the expected area, we're only interested in the maximum activation in a certain local space. Therefore I also conducted the computation that takes a maximum in local and puts 0 to the non-maximum area. The result becomes like the image below.

<img src="./data/images/local_maximization.png" width="500">

Lastly, get the location that has a value above the threshold and it is considered as one of the representatives.

### extraction
In this process, as the name indicates, the candidates are extracted and converted into the original size and rotation based on the score in the previous process. Since we have an access to the detected size and rotation, we can easily extract it and convert it into the desired one. After the extraction and the conversion, you can get the images below. The conversion process includes rotation and pooling computation.

<img src="./data/images/rotated_individual.png" width="100"><img src="./data/images/converted_individual.png" width="100">

### Comparison
This is simplest process. The processed image is compared with all examples and is classified as the most similar example. The final result will be:

<img src="./data/images/result_representation.png" width="500">

---
## Docker Environment
This repository contains docker environment to allow anyone to try my model. To make the execution simple, I created my environment with docker-compose. Please follow the procedure below to build my environment.

1. Go to `Docker/`
2. `docker-compose up -d`
3. Execute the python code denoted in the next section.

---
## How to run the code
It contains 2 functions; `generate_representation.py` and `detection.py`. The former indicates the image with some markers of data and the other is used to detect the markers from a selected image. You're able to control the minimum and maximum size of the marker, the maximum rotation, and the level of noise in both generation and detection process. Please visit the function to further information.  You can run the function with the codes below.
- generation
```bash
docker-compose marker_detection python3 pyfiles/generate_representation.py "./data/representation.png"
```
the argument specifies the path of the target representation.
- detection
```bash
docker-compose marker_detection python3 pyfiles/detection.py "./data/representation.png" "./data/"
```
the arguments indicate the path of the target image and the directory path which will include the result, respectively.


## Contact
Feel free to contact me if you have any questions(<s-inoue-tgz@eagle.sophia.ac.jp>).

