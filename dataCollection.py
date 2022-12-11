import cv2
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pathlib import Path
import math

# Setup and select an alphabet to collect data for
COL_NAMES = ['WRISTX', 'WRISTY',
             'THUMB_CMCX', 'THUMB_CMCY', 'THUMB_MCPX', 'THUMB_MCPY', 'THUMB_IPX', 'THUMB_IPY', 'THUMB_TIPX', 'THUMB_TIPY',
             'INDEX_FINGER_MCPX', 'INDEX_FINGER_MCPY', 'INDEX_FINGER_PIPX', 'INDEX_FINGER_PIPY', 'INDEX_FINGER_DIPX', 'INDEX_FINGER_DIPY', 'INDEX_FINGER_TIPX', 'INDEX_FINGER_TIPY',
             'MIDDLE_FINGER_MCPX', 'MIDDLE_FINGER_MCPY', 'MIDDLE_FINGER_PIPX', 'MIDDLE_FINGER_PIPY', 'MIDDLE_FINGER_DIPX', 'MIDDLE_FINGER_DIPY', 'MIDDLE_FINGER_TIPX', 'MIDDLE_FINGER_TIPY',
             'RING_FINGER_MCPX', 'RING_FINGER_MCPY', 'RING_FINGER_PIPX', 'RING_FINGER_PIPY', 'RING_FINGER_DIPX', 'RING_FINGER_DIPY', 'RING_FINGER_TIPX', 'RING_FINGER_TIPY',
             'PINKY_MCPX', 'PINKY_MCPY', 'PINKY_PIPX', 'PINKY_PIPY', 'PINKY_DIPX', 'PINKY_DIPY', 'PINKY_TIPX', 'PINKY_TIPY',]
SIZE = 1000
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
alphabet = 'T'
count = 0
df = pd.DataFrame(columns=COL_NAMES)

while True:
    # Find one hand using webcam
    success, img = cap.read()
    hands, img = detector.findHands(img)
    processed = 0

    if hands:
        hand = hands[0]
        handType = hand['type']
        lmList = hand['lmList']

        if handType == 'Left' and len(lmList) == 21:

            # Strip the z-axis
            lmList = np.array(lmList)[:, 0:2]

            # Get relative coordinates of each landmark to the wrist
            baseX, baseY = lmList[0][0], lmList[0][1]
            for lm in lmList:
                lm[0] = lm[0] - baseX
                lm[1] = lm[1] - baseY

            # Rotate the coordinates based on the angle between y-axis and point(INDEX_FINGER_MCP)
            point = lmList[5]
            pointX = point[0]
            pointY = point[1]
            if pointY != 0:
                theta = math.atan(pointX/pointY)
            else:
                if pointX > 0:
                    theta = -math.pi/2
                else:
                    theta = math.pi/2

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
            lmList = lmList/maxVal
            processed = 1

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    # Save data
    if key == ord("s") and processed:
        df.loc[f'{count}'] = lmList
        count += 1
        print(count)
        if count == SIZE:
            key = ord('q')
            print(f'The dataset has reached {SIZE} rows.')

    # Quit operation
    if key == ord("q"):
        filepath = Path(f'DATA/{alphabet}.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        break
