import sys
import time
import math
import cv2
import keras
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, 
                            QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox)

SECPERMIN = 60
GAMETIME = 60
PUNCS = [',','.','?','\'','!',' ']
APPSTYLE = '''
    QPushButton{
        background: #262626;
        border-radius: 10px;
        font-family: Arial;
        color: #ffffff;
        font-size: 20px;
        background: #262626;
        padding: 10px 20px 10px 20px;
    }

    QPushButton:hover{
        background: #f73636;
    }

    QLabel{
        font-family: Arial;
        font-size: 20px;
    }
'''

# Get labels
labels = []
with open('Model/labels.txt', 'r') as file:
    for line in file.readlines():
        alphabet = line.rstrip('\n')
        labels.append(alphabet)
print(labels)

class resultDialog(QMessageBox):
    def __init__(self):
        super().__init__()

class videoThread(QThread):
    updateFrame = pyqtSignal(QImage)
    handSign = pyqtSignal(str)
    def __init__(self, parent = None):
        super().__init__()
        self.predictedClass = None

    def run(self):
        self.status = True
        self.cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        model = keras.models.load_model('Model/hand_sign_detection.h5')
        offset = 20
        textHeight = 100
        self.predictedClass = '0'
        while self.status:
            success, img = self.cap.read()
            if success:
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
                        self.predictedClass = labels[np.argmax(prediction)]
                        cv2.rectangle(imgOutput, (x - offset, y), (x + w + offset, y - textHeight), (255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, self.predictedClass, (x + math.ceil(w/2), y - math.ceil(textHeight/4)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)
                else:
                    self.predictedClass = '0'
                
                # Get the latest frame and convert into Image
                imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)

                # Creating and scaling QImage
                h, w, ch = imgOutput.shape
                img = QImage(imgOutput.data, w, h, ch * w, QImage.Format.Format_RGB888)
                scaled_img = img.scaled(600, 450, Qt.AspectRatioMode.KeepAspectRatio)

                # Emit signals
                self.updateFrame.emit(scaled_img)
                self.handSign.emit(self.predictedClass)
        return
    
    def stop(self):
        self.status = False
        self.exit()

class gameThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, parent: None, text):
        super().__init__()
        self.text = text
        self.prediction = None

    def run(self):
        self.status = True
        self.currentText = self.text
        self.gameFinished = False
        self.endtime = False
        textLength = len(self.text)
        currentIndex = 0
        self.count = 0
        self.prediction = '0'
        while self.status and not self.endtime and not self.gameFinished:
            currentLetter = self.text[currentIndex]
            
            # Game completed
            if currentIndex == textLength - 1:
                self.gameFinished = True

            # Skip puctuation
            elif currentLetter in PUNCS:
                currentIndex += 1
                self.currentText = self.text[0:currentIndex] + '|' + self.text[currentIndex:]

            # Match
            elif currentLetter != '0' and currentLetter.lower() == self.prediction.lower():
                currentIndex += 1
                self.count += 1
                self.currentText = self.text[0:currentIndex] + '|' + self.text[currentIndex:]

            # Emit signal
            self.progress.emit(self.currentText)
        return
    
    def stop(self):
        self.status = False
        self.exit()

    def receiveHangSign(self, letter):
        self.prediction = letter

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.widgets = {
            'logo':[],
            'plaintxt1':[],
            'plaintxt2':[],
            'difficulty1':[],
            'difficulty2':[],
            'difficulty3':[],
            'cam':[],
            'time':[],
            'startbtn':[],
            'closebtn':[],
            'gametxt':[],
            'wpm':[]
        }
        
        # Title and dimensions
        self.setWindowTitle("Speedy Hand Sign")
        self.setGeometry(0, 0, 1200, 1400)
        self.difficulty = None

        # Initializing layout
        self.widget = QWidget()
        self.grid = QGridLayout()
        self.widget.setLayout(self.grid)
        self.setCentralWidget(self.widget)

        # Show the first frame
        self.difficultyFrame()

        # Set the style sheet
        self.setStyleSheet(APPSTYLE)

    def difficultyFrame(self):

        # Title
        self.logo = QLabel()
        self.logo.setText('Speedy Hand Sign')
        self.logo.adjustSize()
        self.widgets['logo'].append(self.logo)

        self.plaintxt1 = QLabel()
        self.plaintxt1.setText('Choose a difficulty:')
        self.widgets['plaintxt1'].append(self.plaintxt1)

        # Select a difficulty
        self.difficulty1 = QPushButton('Easy')
        self.difficulty1.clicked.connect(self.setEasyDifficulty)
        self.widgets['difficulty1'].append(self.difficulty1)

        self.difficulty2 = QPushButton('Medium')
        self.difficulty2.clicked.connect(self.setMediumDifficulty)
        self.widgets['difficulty2'].append(self.difficulty2)
        
        self.difficulty3 = QPushButton('Hard')
        self.difficulty3.clicked.connect(self.setHardDifficulty)
        self.widgets['difficulty3'].append(self.difficulty3)
        
        # Grid layout
        self.grid.addWidget(self.widgets['logo'][-1], 0, 1, Qt.AlignmentFlag.AlignCenter)
        self.grid.addWidget(self.widgets['plaintxt1'][-1], 1, 1, Qt.AlignmentFlag.AlignCenter)
        self.grid.addWidget(self.widgets['difficulty1'][-1], 2, 0, Qt.AlignmentFlag.AlignHCenter)
        self.grid.addWidget(self.widgets['difficulty2'][-1], 2, 1, Qt.AlignmentFlag.AlignHCenter)
        self.grid.addWidget(self.widgets['difficulty3'][-1], 2, 2, Qt.AlignmentFlag.AlignHCenter)

    def gameFrame(self):

        # Create a label for the display camera
        self.cam = QLabel(self)
        self.cam.setFixedSize(600, 450)
        self.widgets['cam'].append(self.cam)

        # Thread in charge of updating the image
        self.vth = videoThread(self)
        self.vth.finished.connect(self.close)
        self.vth.updateFrame.connect(self.setImage)
        self.vth.handSign.connect(self.sendHandSign)

        # Thread in charge of the game
        self.gth = gameThread(self, self.text)
        self.gth.finished.connect(self.result)
        self.gth.progress.connect(self.gameText)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.startbtn = QPushButton("Start")
        self.closebtn = QPushButton("Close")
        buttons_layout.addWidget(self.closebtn)
        buttons_layout.addWidget(self.startbtn)
        self.widgets['startbtn'].append(self.startbtn)
        self.widgets['closebtn'].append(self.closebtn)

        # One-minute Timer
        self.time = QLabel('Remaining time: \n')
        self.widgets['time'].append(self.time)

        # Game text
        self.gametxt = QLabel('Current progress: \n')
        self.gametxt.setWordWrap(True)
        self.gametxt.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.widgets['gametxt'].append(self.gametxt)
        
        #Right layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.time)
        rightLayout.addWidget(self.gametxt)

        # Upper layout
        upperLayout = QHBoxLayout()
        upperLayout.addWidget(self.cam)
        upperLayout.addLayout(rightLayout)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(upperLayout)
        layout.addLayout(buttons_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.startbtn.clicked.connect(self.start)
        self.closebtn.clicked.connect(self.kill_thread)
        self.closebtn.clicked.connect(self.close)
        self.closebtn.setEnabled(False)

    def setEasyDifficulty(self):

        # Set difficulty to easy
        self.difficulty = 'easy'
        self.text = self.getText(self.difficulty)
        self.clearWidgets()
        self.gameFrame()
    
    def setMediumDifficulty(self):

        # Set difficulty to medium
        self.difficulty = 'medium'
        self.text = self.getText(self.difficulty)
        self.clearWidgets()
        self.gameFrame()
    
    def setHardDifficulty(self):

        # Set difficulty to Hard
        self.difficulty = 'Hard'
        self.text = self.getText(self.difficulty)
        self.clearWidgets()
        self.gameFrame()

    def clearWidgets(self):

        for widget in self.widgets:

            # Hide all the widgets
            if self.widgets[widget] != []:
                self.widgets[widget][-1].hide()
            
            # Remove them from the dictionary
            for i in range(0, len(self.widgets[widget])):
                self.widgets[widget].pop()

    def getText(self, difficulty):

        # Obtain text based on the selected difficulty
        text = ''
        with open('text/' + difficulty + '.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                text += line.strip() + ' '
        return text

    def startTimer(self):

        # Start a 60 second countdown timer
        self.timeLeft = GAMETIME
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timeoutTimer)
        self.timer.start(1000)

        self.updateTimer()

    def timeoutTimer(self):

        # Countdown
        self.timeLeft -= 1
        self.updateTimer()
        if self.timeLeft == 0:
            self.gth.endtime = False
            self.result()
            print('Time\'s up')

    def updateTimer(self):

        # Show time
        self.time.setText('Remaining time: ' + str(self.timeLeft) + 'seconds left')

    def result(self):

        # Stop the timer and calculate LPM
        self.timer.stop()
        lpm = SECPERMIN*self.gth.count/(GAMETIME-self.timeLeft)
        
        # Show dialog
        self.rdlg = resultDialog()
        self.rdlg.setWindowTitle("Result")
        self.rdlg.setText("Your result is:\nLPM: " + "%.2f" % lpm)
        self.rdlg.addButton(QMessageBox.StandardButton.Close)
        button = self.rdlg.exec()

        # Close the app
        if button == QMessageBox.StandardButton.Close:
            print("Close!")
            self.kill_thread()
            self.close()

    @pyqtSlot()
    def kill_thread(self):
        print("Finishing...")
        cv2.destroyAllWindows()
        self.vth.stop()
        self.gth.stop()

        # Give time for the threads to finish
        time.sleep(1)

    @pyqtSlot()
    def start(self):
        print("Starting...")
        self.closebtn.setEnabled(True)
        self.startbtn.setEnabled(False)
        self.vth.start()
        self.gth.start()
        self.startTimer()


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.cam.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def sendHandSign(self, handSign):
        self.gth.receiveHangSign(handSign)

    @pyqtSlot(str)
    def gameText(self):
        self.gametxt.setText('Current progress: ' + self.gth.currentText)

def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__": 
    main()