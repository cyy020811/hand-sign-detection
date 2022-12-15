# Hand Sign Detection

## Introduction
The purpose of this project is to build a program that can detect alphabetical signs of American Sign Language (ASL) from the user using a webcam. The project is broken down into three tasks: data collection and preprocessing, modeling, and testing. The key concept of this project is to apply the `MediaPipe Hand` library to capture the coordinates of knuckles, wrists, and other distinguishable spots of a human palm known as hand landmarks as data, then use the dataset to train a feedforward artificial neural network model using `Keras API`. To demonstrate the result, the project also includes an interactive application that allows the user to test out the performance of the model and have a bit of fun with a speed spelling game! The following paragraphs will explain how to install and run each python file and the details of the project. 
 
### Task 1: Data Collection and Preprocessing
The hand landmarks are represented in 2D coordinates based on their absolute positions within the webcam. Hence, the coordinate of `WRIST` is used as the origin to obtain the relative coordinates of other landmarks in order to ensure that all hand signs can be detected regardless of the hand's position within the camera range. Also, considering that the user's hand might rotate, the coordinates are rotated according to the degree between the line connecting `WRIST` and `INDEX_FINGER_MCP` and the y-axis. Lastly, divide all coordinates by absolute value of the largest x- or y-coordinate to normalise the data.

![image](hand_landmarks.png)

#### Instructions
Run the `dataCollection.py` file to collect dataset for modeling. The collected data is stored inside the `Data` folder as csv file with respect to the name of the file.  
#### Controls:  
- Press `s` on the keyboard to save data
- Press `q` on the keyboard to exit the program  

### Task 2: Modeling  
Run the `model.py` file to start modeling with the dataset collected previously. The model is saved inside the `Model` folder.  

### Task 3: Testing
To be continued...
