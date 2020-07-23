# default parameters for octo-home-office application 

# import necessary files 
import numpy as np
import dlib
from src.functions import ref3DModel
from scipy.spatial import distance as dist
from imutils import face_utils

# tilt params
face3Dmodel = ref3DModel()
TILT_COUNTER = 0
TILT_ALARM = False

# absence detection params
ALARM_ABSENSE = False
ABSENSE_COUNTER = 0

# drowness detection params
EYE_AR_THRESH = 0.35 # aspect ratio for a blink
EYE_AR_CONSEC_FRAMES = 45 # nbr of consecutive frames to set the flag
COUNTER = 0
ALARM_ON_SLEEP = False

# load dlib's HOG-based frontal face detector  
detector = dlib.get_frontal_face_detector()

# grab the indexes of the facial landmarks for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]