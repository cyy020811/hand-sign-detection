import cv2
import keras
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Setup

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = keras.models.load_model('Model/hand_sign_detection.h5')
offset = 20
textHeight = 100

# Get labels
labels = []
file = open('Model/labels.txt', 'r')
for line in file.readlines():
    label = line.rstrip('\n')
    labels.append(label)
print(labels)
file.close()

while True:
    # Find one hand using webcam
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]

        # Get information of the hand img
        x, y, w, h = hand['bbox']
        handType = hand['type']
        lmList = hand['lmList']

        if len(lmList) == 21:

            # Strip the z-axis
            lmList = np.array(lmList)[:, 0:2]

            # Get relative coordinates of each landmark to the wrist
            baseX, baseY = lmList[0][0], lmList[0][1]
            for lm in lmList:
                lm[0] = lm[0] - baseX
                lm[1] = lm[1] - baseY

                # Flip x-coordinates if the user uses right hand
                if handType == 'Right':
                    lm[0] = -lm[0]

            # Rotate the coordinates based on the angle between y-axis and point(INDEX_FINGER_MCP)
            point = lmList[5]
            pointX = point[0]
            pointY = point[1]
            if pointY != 0:
                theta = math.atan(pointX / pointY)
            else:
                if pointX > 0:
                    theta = -math.pi / 2
                else:
                    theta = math.pi / 2

            for lm in lmList:
                lm[0] = lm[0] * math.cos(theta) - lm[1] * math.sin(theta)
                lm[1] = lm[1] * math.cos(theta) + lm[0] * math.sin(theta)

            # Check if the hand is upside down
            if pointY > 0:
                for lm in lmList:
                    lm[0] = -lm[0]
                    lm[1] = -lm[1]

            # Turn the array into an 1-d array
            lmList = lmList.flatten()

            # Get the maximum absolute value of the array to normalise the coordinates
            maxVal = max(abs(lmList))
            lmList = lmList / maxVal
            lmList = lmList.reshape(1, 42)

            # Get and show prediction
            prediction = model.predict([lmList])
            predictedClass = labels[np.argmax(prediction)]
            print(predictedClass)
            cv2.rectangle(imgOutput, (x - offset, y), (x + w + offset, y - textHeight), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, predictedClass, (x + math.ceil(w/2), y - math.ceil(textHeight/4)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)
    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
