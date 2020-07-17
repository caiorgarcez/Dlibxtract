# OctoFace facial landmarks for privacy home office productivity monitoring 
# author: caiocrgarcez
# email: caio.garcez1@gmail.com

# version v3: implementation of absence detection. 

# Necessary packages
from imutils import face_utils
import numpy as np
import argparse
import dlib
import imutils
import cv2
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import time

# parse params. USAGE: python octo-lm-caio-v1.py -sp /landmark... -isrc 0 
ap = argparse.ArgumentParser(description="Octoface facial landmarks for privacy home office productivity monitoring")
ap.add_argument("-sp", "--shape-predictor", required=True,
	help="path to facial landmark predictor. Must be 68-pionts.") 
ap.add_argument("-isrc", "--image-source", required=True,
	help="path to input image")
ap.add_argument("-dt", "--drowness-detection", required=True,
	help="activate drowness detection.")
ap.add_argument("-ad", "--absense-detection", required=True,
	help="activate absense detection.")
args = vars(ap.parse_args())

# functions 
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)

def ref2dImagePoints(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)

def fcameraMatrix(fl, center):
    mat = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float)


# tilt params
face3Dmodel = ref3DModel()
TILT_COUNTER = 0
TILT_ALARM = False

# absence detection params
ALARM_ABSENSE = False
ABSENSE_COUNTER = 0

# drowness detection params
EYE_AR_THRESH = 0.3 # aspect ratio for a blink
EYE_AR_CONSEC_FRAMES = 48 # nbr of consecutive frames to set the flag
COUNTER = 0
ALARM_ON_SLEEP = False

# dlib's HOG-based frontal face detector  # (TODO: ResNet-50 or 34 option for face detection)
detector = dlib.get_frontal_face_detector()

# dlib's trained facial landmark predictor  # (TODO: Options for other N-point detectors)
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Main program
if int(args["image_source"]) == 0: 
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    
    while True:
        # read the next frame from the video stream, resize it, and convert it to grayscale for faster processing
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image without upsampling
        rects = detector(gray, 0) 
        
        # create a mask for privacy
        mask = np.zeros(gray.shape, np.uint8)

        # loop over the face detections # (TODO: function to select the bbox with the largest area)
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape_dlib = shape
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # (DEBUG: display roi in original frame) 
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(gray, "Face", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            refImgPts = ref2dImagePoints(shape_dlib)
            height = 500
            width = 500

            focalLength = 1 * width
            cameraMatrix = fcameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(mask, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            if angles[1] < -15:
                TILT_COUNTER += 1

                if TILT_COUNTER >= EYE_AR_CONSEC_FRAMES*2:
                        # if the alarm is not on, turn it on
                        if not TILT_ALARM:
                            TILT_ALARM = True

                        # draw an alarm on the frame
                        cv2.putText(mask, " Alerta: DESATENTO!!", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            elif angles[1] > 15:
                TILT_COUNTER += 1

                if TILT_COUNTER >= EYE_AR_CONSEC_FRAMES*2:
                        # if the alarm is not on, turn it on
                        if not TILT_ALARM:
                            TILT_ALARM = True

                        # draw an alarm on the frame
                        cv2.putText(mask, " Alerta: DESATENTO!!", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            else:
                TILT_COUNTER = 0
                TILT_ALARM = False

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the mask
            for (x1, y1) in shape:
                cv2.circle(mask, (x1, y1), 1, (255, 255, 255), -1)
                cv2.circle(gray, (x1, y1), 1, (255, 255, 255), -1)
            
            if args["drowness_detection"] == '1': 
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(gray, [leftEyeHull], -1, (255, 255, 255), 1)
                cv2.drawContours(gray, [rightEyeHull], -1, (255, 255, 255), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not ALARM_ON_SLEEP:
                            ALARM_ON_SLEEP = True

                        # draw an alarm on the frame
                        cv2.putText(mask, " Alerta: SONO!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0
                    ALARM_ON_SLEEP = False

                # thresholds and frame counters
                # cv2.putText(mask, "R: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


        if not rects:

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)

            start = time.time()

            # thresholds and frame counters
            cv2.putText(mask, "Alerta: AUSENTE!", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(mask, f"Hora local: {current_time} !", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            
        # show the output frame
        cv2.imshow("Frame", gray)
        cv2.imshow("Filtered", mask)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanu
    cv2.destroyAllWindows()
    vs.stop()


# else:

#     # load the input image, resize it, and convert it to grayscale for faster processing
#     image = cv2.imread(args["image"])
#     image = imutils.resize(image, width=500)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




