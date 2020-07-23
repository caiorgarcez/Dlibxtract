# dlibxtract webcam real time processing code 

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

# custom packages
from src.functions import eye_aspect_ratio, ref2dImagePoints, ref3DModel, fcameraMatrix
from src.params import *

# parse params. USAGE: python main.py -sp /landmark... -isrc 0 -dt 1 -ad 1
ap = argparse.ArgumentParser(description="Implementation of general facial landmarks features based on dlib's facial recognition library")
ap.add_argument("-sp", "--shape-predictor", required=True,
	help="path to facial landmark predictor. Must be 68-pionts.") 
ap.add_argument("-isrc", "--image-source", required=True,
	help="path to input image")
ap.add_argument("-dt", "--drowsiness-detection", required=True,
	help="activate drowsiness detection.")
ap.add_argument("-ad", "--absense-detection", required=True,
	help="activate absense detection.")
args = vars(ap.parse_args())

# load dlib's trained facial landmark predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# Main program
if int(args["image_source"]) == 0: 
    # initialize videostream thread
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        time.sleep(5)
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
                        cv2.putText(mask, "Alert: Not focused!", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            elif angles[1] > 15:
                TILT_COUNTER += 1

                if TILT_COUNTER >= EYE_AR_CONSEC_FRAMES*2:
                        # if the alarm is not on, turn it on
                        if not TILT_ALARM:
                            TILT_ALARM = True

                        # draw an alarm on the frame
                        cv2.putText(mask, "Alert: Not focused!", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            else:
                TILT_COUNTER = 0
                TILT_ALARM = False

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the mask
            for (x1, y1) in shape:
                cv2.circle(mask, (x1, y1), 1, (255, 255, 255), -1)
                cv2.circle(gray, (x1, y1), 1, (255, 255, 255), -1)
            
            if args["drowsiness_detection"] == '1': 
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
                        cv2.putText(mask, " Alert: sleep!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0
                    ALARM_ON_SLEEP = False

                # thresholds and frame counters. Uncomment for trim.
                # cv2.putText(mask, "R: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Display a text in case of no dectection
        if not rects:
            # thresholds and frame counters
            cv2.putText(mask, "No detection", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        # show the output frame
        cv2.imshow("Frame", gray)
        cv2.imshow("Filtered", mask)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
