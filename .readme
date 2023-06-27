# Basketball Shot Analysis


This repository contains the source code for our project: a tool for basketball shot analysis using computer vision techniques. The primary goal of our project is to assist basketball players, both professional and amateur, in improving their shooting skills.

We developed this tool to extract important metrics from a player's shooting action, such as grip, eye focus, release, follow-through, arc, body alignment, footwork and positioning, elbow tightness, and knee bend. These metrics are crucial for achieving a good basketball shot. Using our tool, players can compare their shots to those of professional players and gain insights into areas of improvement. The tool also includes a web interface for uploading videos and obtaining shot analysis information.

## Features

1. Basketball and Rim Detection: Using YOLO object detection model, the application can identify the basketball and the rim within each frame of a video. This provides the basis for the rest of the shot analysis. We trained and fine-tuned the YOLO model using a dataset from Roboflow.

2. Body Landmarks and Angle Detection: The application utilizes the MediaPipe library to identify the player's body landmarks necessary for the shot analysis. It can detect the main landmarks on the shooter’s body, such as elbows, hands, knees, shoulders, feet, and head. It can also calculate angles, like knee angle and elbow angle, crucial for shot performance.

3. Comparative Analysis: Users can upload two videos - one of their shot and another of a professional player. The tool will generate a table comparing the key metrics between the two shots.


## Installation

To set up the project locally, please follow the instructions below:
1. Clone the repository to your local machine using the git clone command.
2. Install the necessary dependencies as listed in the *requirements.txt* file using pip:
```sh
pip3 install -r requirements.txt
```

3. Run the application using the instructions provided in the 'Running the Application' section.

## Running the Applucation
Please follow the instructions below to run the application:

1. Open the terminal in the project directory.
2. Run the application by doing the followning:
```sh
cd source
streamlit run user_interface.py 
```

## Dataset

We used an open-source dataset from Roboflow that includes approximately 3,000 annotated images of basketballs, people, and rims. The dataset was used to train the YOLO model for object detection.

## Training the Model

We trained the YOLOv5 model with the following hyperparameters:

- Momentum: 0.95
- Weight Decay: 0.0005
- Epochs: 40
- Batch Size: 16
- Optimizer: SGD
- IOU threshold: 0.2 (for evaluation)


There were also 3 warmup epochs with a momentum of 0.8.
The results of training can be seen in the weights/metrics folder, where we can see relevant plots like the confusion matrix and F1curve.

## Future Work

The current version of the application only analyzes shots in isolation. In future versions, we aim to incorporate player tracking to analyze shots in the context of a game scenario. We also plan to include other metrics such as player speed, dribbling skills, and defensive skills.

## Contributors
This project was a joint effort by Simon and Gautam. We welcome any contributions or suggestions to improve the project.

## References
The details and references used in this project can be found in the "report.pdf" file attached.



   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>