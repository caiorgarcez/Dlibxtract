# dlibxtract
Implementation of general facial landmarks features based on dlib's facial recognition library.

-----

### 1. Install the environment

Packages and other requirements to run `dlibxtract` is provided for conda in `requirements.yml` and for pip in `requirements.txt`.

for conda run: 
`$ conda env create -f requirements.yml`

for pip run:
`$ pip install -r requirements.txt` 

-----

### 2. Usage and features:

`-sp` Path to facial landmark predictor. Must be 68-pionts.  
`-iscr` Path of the video file.  path to input image.  If the video source comes from a webcam set: <0>
`-dt` Activate drowsiness detection.
`-ad` Activate absense detection.

Example: `$ python main.py -sp /landmark... -isrc 0 -dt 1 -ad 1` 

----- 

### 3. Observations:

The code for this project is based in: [[1]](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/), [[2]](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
